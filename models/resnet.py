from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def conv5x5(in_planes: int, out_planes: int) -> nn.Conv2d:
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=2)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, 
                    bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_fordecoder(in_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, in_planes//2, kernel_size=1, stride=1, bias=False)

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride: int = 1, downsample: Optional[nn.Module] = None, 
                 groups: int = 1, base_width: int = 64, dilation: int = 1) -> None: 
        super().__init__()
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) 
        self.bn2 = norm_layer(width) 
        self.conv3 = conv1x1(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride 

    def forward(self, x: Tensor) -> Tensor: 
        identity = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) 
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None: 
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out 
    
class ResNet50Encoder(nn.Module): 
    def __init__(self, block: Type[Union[Bottleneck]], layers: List[int] = [3, 4, 6, 3], 
                 num_classes: int = 1000, zero_init_residual: bool = False, groups: int = 1,
                 width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None: 
        super().__init__()
        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.inplanes = 64 
        self.dilation = 1 
        if replace_stride_with_dilation is None: 
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups 
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
    
    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int, 
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # add a conv1x1 layer to downsample the input in case if the sizes of input and output from the resnet internal layers are different
        if stride!=1 or self.inplanes != planes*block.expansion: 
            downsample = nn.Sequential(conv1x1(self.inplanes, planes*block.expansion, stride),
                                        norm_layer(planes*block.expansion))
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample,self.groups,self.base_width,previous_dilation))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,planes,groups=self.groups,base_width=self.base_width,dilation=self.dilation))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)           # 480x480x3 -> 240x240x64
        x = self.bn1(x)             # 240x240x64
        x = self.relu(x)            # 240x240x64
        x = self.maxpool(x)         # 240x240x64 -> 120x120x64
        
        x1 = self.layer1(x)         # 120x120x64 -> 120x120x256
        x2 = self.layer2(x1)        # 120x120x256 -> 60x60x512
        x3 = self.layer3(x2)        # 60x60x512 -> 30x30x1024
        x4 = self.layer4(x3)        # 30x30x1024 -> 15x15x2048

        return x1, x2, x3, x4

class ResNetUpProjection(nn.Module):
    def __init__(self, inplanes: int, outplanes: int) -> None:
        super().__init__()
        self.unpool = nn.Upsample(scale_factor=2)#(kernel_size=2, stride=2)
        self.conv1 = conv5x5(inplanes, outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.conv3 = conv5x5(outplanes, outplanes)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.unpool(x)          # 15x15x1024 -> 30x30x1024
        y = self.conv1(x)           # 30x30x1024 -> 30x30x512
        y = self.relu(y)            # 30x30x512
        y = self.conv2(y)           # 30x30x512 -> 30x30x512
        y = y + self.conv3(x)       # 30x30x512
        return self.relu(y)         # 30x30x512

class ResNet50CSL(nn.Module):
    def __init__(self, num_classes, num_joints) -> None: 
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.resnet_encoder = ResNet50Encoder(Bottleneck, [3, 4, 6, 3])
        self.conv1_dec = conv1x1_fordecoder(2048)
        self.decoder_layer1 = ResNetUpProjection(1024, 512)
        self.conv2_dec = conv1x1_fordecoder(1024)
        self.decoder_layer2 = ResNetUpProjection(512, 256)
        self.conv3_dec = conv1x1_fordecoder(512)
        self.decoder_layer3 = ResNetUpProjection(256, 128)
        self.conv4_dec = conv1x1_fordecoder(256)
        self.decoder_layer4 = ResNetUpProjection(128, 64)
        self.seg_layer = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.seg_head = nn.Softmax(dim=1)
        self.decoder_layer5 = conv1x1(64, 32)
        self.local_head = nn.Conv2d(32+self.num_classes, self.num_joints, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: Tensor) -> Tensor:
        x1, x2, x3, x4 = self.resnet_encoder(x)     # 480x480x3 -> 15x15x2048
        import pdb; pdb.set_trace()
        y = self.conv1_dec(x4)                      # 15x15x2048 -> 15x15x1024
        y = self.decoder_layer1(y)                  # 15x15x1024 -> 30x30x512
        y += self.relu(self.conv2_dec(x3))          # 30x30x512
        y = self.decoder_layer2(y)                  # 30x30x512 -> 60x60x256
        y += self.relu(self.conv3_dec(x2))          # 60x60x256
        y = self.decoder_layer3(y)                  # 60x60x256 -> 120x120x128
        y += self.relu(self.conv4_dec(x1))          # 120x120x128
        y = self.decoder_layer4(y)                  # 120x120x128 -> 240x240x64
        y_seg = self.seg_layer(y)                   # 240x240x64 -> 240x240xnum_classes
        y = self.decoder_layer5(y)                  # 240x240x64 -> 240x240x32
        y = torch.concat((y, y_seg), dim=1)         # 240x240x32+num_classes
        y = self.local_head(y)                      # 240x240x32+num_classes -> 240x240xnum_joints
        y_seg = self.seg_head(y_seg)                # 240x240xnum_joints
        y_seg = F.upsample_bilinear(y_seg, scale_factor=2) # 240x240xnum_classes -> 480x480xnum_classes
        y = F.upsample_bilinear(y, scale_factor=2) # 240x240xnum_joints -> 480x480xnum_joints
        return y_seg, y
