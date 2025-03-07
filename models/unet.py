import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv = UNetBlock(2*out_channels, out_channels)
    
    def forward(self, x: Tensor, x_enc: Tensor) -> Tensor:
        x = self.unpool(x)
        x = self.upconv(x)
        x = torch.cat([x, x_enc], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, output_channels: int, input_channels: int = 3):
        super().__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.conv1 = UNetBlock(self.input_channels, 64)
        self.conv2 = UNetBlock(64, 128)
        self.conv3 = UNetBlock(128, 256)
        self.conv4 = UNetBlock(256, 512)
        self.conv5 = UNetBlock(512, 1024)
        self.conv6 = UNetBlock(1024, 2048)
        self.upconv1 = UpConv(2048, 1024)
        self.upconv2 = UpConv(1024, 512)
        self.upconv3 = UpConv(512, 256)
        self.upconv4 = UpConv(256, 128)
        self.upconv5 = UpConv(128, 64)
        self.outconv = nn.Conv2d(64, self.output_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        y = self.conv6(x5)
        y = self.upconv1(y, x5)
        y = self.upconv2(y, x4)
        y = self.upconv3(y, x3)
        y = self.upconv4(y, x2)
        y = self.upconv5(y, x1)
        y = self.outconv(y)
        return y

class KurmannEtAl2017Net(nn.Module):
    def __init__(self, num_classes: int, num_joints: int, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.output_channels = num_classes*num_joints
        self.input_channels = input_channels
        self.conv1 = UNetBlock(self.input_channels, 64)
        self.conv2 = UNetBlock(64, 128)
        self.conv3 = UNetBlock(128, 256)
        self.conv4 = UNetBlock(256, 512)
        self.conv5 = UNetBlock(512, 1024)
        self.conv6 = UNetBlock(1024, 2048)
        self.conv_enc = nn.Conv2d(2048, 128, 3, padding=1)
        self.classifier_layer1 = nn.Linear(15*15*128, 512)
        self.classifier_layer2 = nn.Linear(512, 256)
        self.classifier_layer3 = nn.Linear(256, self.num_classes)
        self.upconv1 = UpConv(2048, 1024)
        self.upconv2 = UpConv(1024, 512)
        self.upconv3 = UpConv(512, 256)
        self.upconv4 = UpConv(256, 128)
        self.upconv5 = UpConv(128, 64)
        self.outconv = nn.Conv2d(64, self.output_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)          # 480x480x3 -> 480x480x64
        x2 = F.max_pool2d(x1, 2)    # 480x480x64 -> 240x240x64
        x2 = self.conv2(x2)         # 240x240x64 -> 240x240x128
        x3 = F.max_pool2d(x2, 2)    # 240x240x128 -> 120x120x128
        x3 = self.conv3(x3)         # 120x120x128 -> 120x120x256
        x4 = F.max_pool2d(x3, 2)    # 120x120x256 -> 60x60x256
        x4 = self.conv4(x4)         # 60x60x256 -> 60x60x512
        x5 = F.max_pool2d(x4, 2)    # 60x60x512 -> 30x30x512
        x5 = self.conv5(x5)         # 30x30x512 -> 30x30x1024
        x_enc = F.maxpool2d(x5, 2)  # 30x30x1024 -> 15x15x1024
        x_enc = self.conv6(x_enc)   # 15x15x1024 -> 15x15x2048
        y = self.upconv1(x_enc, x5)     # 15x15x2048 -> 30x30x1024
        y = self.upconv2(y, x4)     # 30x30x1024 -> 60x60x512
        y = self.upconv3(y, x3)     # 60x60x512 -> 120x120x256
        y = self.upconv4(y, x2)     # 120x120x256 -> 240x240x128
        y = self.upconv5(y, x1)     # 240x240x128 -> 480x480x64
        y = self.outconv(y)         # 480x480x64 -> 480x480xnum_classes*num_joints

        x_c = self.conv_enc(x_c)            # 15x15x2048 -> 15x15x128
        x_c = torch.flatten(x_c, 1)         # 15x15x128 -> 15*15*128
        x_c = F.relu(x_c)                   # 15*15*128
        x_c = self.classifier_layer1(x_c)   # 15*15*128 -> 512
        x_c = F.relu(x_c)                   # 512
        x_c = self.classifier_layer2(x_c)   # 512 -> 256
        x_c = F.relu(x_c)                   # 256
        x_c = self.classifier_layer3(x_c)   # 256 -> num_classes
        x_c = F.softmax(x_c, dim=1)
        return y, x_enc
