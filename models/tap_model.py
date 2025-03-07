import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

class Conv2dReLU(nn.Module):
    """
    [Conv2d(in_channels, out_channels, kernel),
    BatchNorm2d(out_channels),
    ReLU,]
    """
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, bn=False):
        super(Conv2dReLU, self).__init__()
        modules = OrderedDict()
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if bn:
            modules['bn'] = nn.BatchNorm2d(out_channels)
        modules['relu'] = nn.ReLU(inplace=True)
        self.l = nn.Sequential(modules)

    def forward(self, x):
        x = self.l(x)
        return x
        

class UNetModule(nn.Module):
    """

    [Conv2dReLU(in_channels, out_channels, 3),
    Conv2dReLU(out_channels, out_channels, 3)]
    """
    def __init__(self, in_channels, out_channels, padding=1, bn=False):
        super(UNetModule, self).__init__()
        self.l = nn.Sequential(OrderedDict([
            ('conv1', Conv2dReLU(in_channels, out_channels, 3, padding=padding, bn=bn)),
            ('conv2', Conv2dReLU(out_channels, out_channels, 3, padding=padding, bn=bn))
            ]))

    def forward(self, x):
        x = self.l(x)
        return x

class Interpolate(nn.Module):
    """
    Wrapper function of interpolate/UpSample Module
    """
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.fn = lambda x: nn.functional.interpolate(x, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

    def forward(self, x):
        return self.fn(x)

class DecoderModule(nn.Module):
    """
    DecoderModule for UNet11, UNet16
    
    Upsample version:
    [Interpolate(scale_factor, 'bilinear'),
    Con2dReLU(in_channels, mid_channels),
    Conv2dReLU(mid_channels, out_channels),
    ]

    DeConv version:
    [Con2dReLU(in_channels, mid_channels),
    ConvTranspose2d(mid_channels, out_channels, kernel=4, stride=2, pad=1),
    ReLU
    ]
    """
    def __init__(self, in_channels, mid_channels, out_channels, upsample=True):
        super(DecoderModule, self).__init__()
        if upsample:
            modules = OrderedDict([
                ('interpolate', Interpolate(scale_factor=2, mode='bilinear', 
                    align_corners=False)),
                ('conv1', Conv2dReLU(in_channels, mid_channels)),
                ('conv2', Conv2dReLU(mid_channels, out_channels))
                ])
        else:
            modules = OrderedDict([
                ('conv', Conv2dReLU(in_channels, mid_channels)),
                ('deconv', nn.ConvTranspose2d(mid_channels, 
                    out_channels, kernel_size=4, stride=2, padding=1)),
                ('relu', nn.ReLU(inplace=True))
                ])
        self.l = nn.Sequential(modules)

    def forward(self, x):
        return self.l(x)

class AttentionModule(nn.Module):
    """
    attention module:
    incorporating attention map and features
    """
    def __init__(self, in_channels, out_channels, scale_factor, bn=False):
        super(AttentionModule, self).__init__()
        self.scale_factor = scale_factor
        self.downsample = Interpolate(scale_factor=scale_factor,\
             mode='bilinear', align_corners=False)

        self.firstconv = Conv2dReLU(in_channels, out_channels, bn=bn)
        # self-learnable attention map
        self.learnable_attmap = nn.Sequential(
            Conv2dReLU(out_channels, 1, 1, padding=0, bn=bn),
            nn.Sigmoid()
            )

    
    def forward(self, x, attmap, **kwargs):
        if self.scale_factor != 1:
            attmap = self.downsample(attmap)
        x = self.firstconv(x)
        output =  x + (x * attmap)
        attmap_learned = self.learnable_attmap(output)
        return output, attmap_learned



class TAPNet(nn.Module):
    """docstring for TAPNet"""
    def __init__(self, in_channels, num_classes, bn=False):
        super(TAPNet, self).__init__()

        # half the kernel size
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = UNetModule(in_channels, 32, bn=bn)
        self.conv2 = UNetModule(32, 64, bn=bn)
        self.conv3 = UNetModule(64, 128, bn=bn)
        self.conv4 = UNetModule(128, 256, bn=bn)
        self.center = UNetModule(256, 512, bn=bn)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)
        self.up4 = UNetModule(512 + 256, 256)
        self.up3 = UNetModule(256 + 128, 128)
        self.up2 = UNetModule(128 + 64, 64)
        self.up1 = UNetModule(64 + 32, 32)

        self.att4 = AttentionModule(512 + 256, 512 + 256, 1/8, bn=bn)
        self.att3 = AttentionModule(256 + 128, 256 + 128, 1, bn=bn)
        self.att2 = AttentionModule(128 + 64, 128 + 64, 1, bn=bn)
        self.att1 = AttentionModule(64 + 32, 64 + 32, 1, bn=bn)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)

        
    def forward(self, x, attmap, **kwargs):
        conv1 = self.conv1(x) # 32
        conv2 = self.conv2(self.maxpool(conv1)) # 64
        conv3 = self.conv3(self.maxpool(conv2)) # 128
        conv4 = self.conv4(self.maxpool(conv3)) # 256
        center = self.center(self.maxpool(conv4)) # 512
        
        att4, attmap4 = self.att4(torch.cat([conv4, self.upsample(center)], 1), attmap)
        up4 = self.up4(att4)
        # up4 = self.up4(torch.cat([conv4, self.upsample(center)], 1))
        att3, attmap3 = self.att3(torch.cat([conv3, self.upsample(up4)], 1), self.upsample(attmap4))
        up3 = self.up3(att3)
        att2, attmap2 = self.att2(torch.cat([conv2, self.upsample(up3)], 1), self.upsample(attmap3))
        up2 = self.up2(att2)
        att1, attmap1 = self.att1(torch.cat([conv1, self.upsample(up2)], 1), self.upsample(attmap2))
        up1 = self.up1(att1)

        output = self.final(up1)
        # if output attmap, occupy too much space
        return output


class TAPNet11(nn.Module):
    """
    docstring for TAPNet11
    use VGG11 as encoder
    """
    def __init__(self, in_channels, num_classes, pretrained=True, bn=False, upsample=False):
        super(TAPNet11, self).__init__()
        self.num_classes = num_classes
        self.vgg11 = models.vgg11(pretrained=pretrained).features
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = self.vgg11[0:2]
        self.conv2 = self.vgg11[3:5]
        self.conv3 = self.vgg11[6:10]
        self.conv4 = self.vgg11[11:15]
        self.conv5 = self.vgg11[16:20]
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(256 + 512, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(256 + 512, 512, 128, upsample=upsample)
        self.dec3 = DecoderModule(128 + 256, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(64 + 128, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(32 + 64, 32)

        self.att5 = AttentionModule(256 + 512, 256 + 512, 1/16, bn=bn)
        self.att4 = AttentionModule(256 + 512, 256 + 512, 1, bn=bn)
        self.att3 = AttentionModule(128 + 256, 128 + 256, 1, bn=bn)
        self.att2 = AttentionModule(64 + 128, 64 + 128, 1, bn=bn)
        self.att1 = AttentionModule(32 + 64, 32 + 64, 1, bn=bn)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)


    def forward(self, x, attmap, **kwargs):
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.maxpool(conv1)) # 128
        conv3 = self.conv3(self.maxpool(conv2)) # 256
        conv4 = self.conv4(self.maxpool(conv3)) # 512
        conv5 = self.conv5(self.maxpool(conv4)) # 512
        center = self.center(self.maxpool(conv5)) # 256

        
        att5, attmap5 = self.att5(torch.cat([center, conv5], 1), attmap)
        dec5 = self.dec5(att5)
        # dec5 = self.dec5(torch.cat([center, conv5], 1))
        att4, attmap4 = self.att4(torch.cat([dec5, conv4], 1), self.upsample(attmap5))
        dec4 = self.dec4(att4)
        # dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        att3, attmap3 = self.att3(torch.cat([dec4, conv3], 1), self.upsample(attmap4))
        dec3 = self.dec3(att3)
        att2, attmap2 = self.att2(torch.cat([dec3, conv2], 1), self.upsample(attmap3))
        dec2 = self.dec2(att2)
        att1, attmap1 = self.att1(torch.cat([dec2, conv1], 1), self.upsample(attmap2))
        dec1 = self.dec1(att1)
        # output = self.final(dec1)
        if self.num_classes>1:
            output = F.log_softmax(self.final(dec1), dim=1)
        else:
            output = self.final(dec1)
        return output


class TAPNet16(nn.Module):
    """
    docstring for TAPNet16
    use VGG16 as encoder
    """
    def __init__(self, in_channels, num_classes, pretrained=False, bn=False, upsample=True):
        super(TAPNet16, self).__init__()
        self.num_classes = num_classes
        self.vgg16 = models.vgg16(pretrained=pretrained).features
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv1 = self.vgg16[0:4]
        self.conv2 = self.vgg16[5:9]
        self.conv3 = self.vgg16[10:16]
        self.conv4 = self.vgg16[17:23]
        self.conv5 = self.vgg16[24:30]
        self.upsample = Interpolate(scale_factor=2,\
             mode='bilinear', align_corners=False)

        self.center = DecoderModule(512, 512, 256, upsample=upsample)
        self.dec5 = DecoderModule(256 + 512, 512, 256, upsample=upsample)
        self.dec4 = DecoderModule(256 + 512, 512, 256, upsample=upsample)
        self.dec3 = DecoderModule(128 + 256, 256, 64, upsample=upsample)
        self.dec2 = DecoderModule(64 + 128, 128, 32, upsample=upsample)
        self.dec1 = Conv2dReLU(32 + 64, 32)

        self.att5 = AttentionModule(256 + 512, 256 + 512, 1/16, bn=bn)
        self.att4 = AttentionModule(256 + 512, 256 + 512, 1, bn=bn)
        self.att3 = AttentionModule(256 + 256, 128 + 256, 1, bn=bn)
        self.att2 = AttentionModule(64 + 128, 64 + 128, 1, bn=bn)
        self.att1 = AttentionModule(32 + 64, 32 + 64, 1, bn=bn)

        # return logits
        self.final = nn.Conv2d(32, num_classes, 1)



    def forward(self, x, attmap, **kwargs):
        conv1 = self.conv1(x) # 64
        conv2 = self.conv2(self.maxpool(conv1)) # 128
        conv3 = self.conv3(self.maxpool(conv2)) # 256
        conv4 = self.conv4(self.maxpool(conv3)) # 512
        conv5 = self.conv5(self.maxpool(conv4)) # 512

        center = self.center(self.maxpool(conv5)) # 256

        att5, attmap5 = self.att5(torch.cat([center, conv5], 1), attmap)
        dec5 = self.dec5(att5)
        att4, attmap4 = self.att4(torch.cat([dec5, conv4], 1), self.upsample(attmap5))
        dec4 = self.dec4(att4)
        att3, attmap3 = self.att3(torch.cat([dec4, conv3], 1), self.upsample(attmap4))
        dec3 = self.dec3(att3)
        att2, attmap2 = self.att2(torch.cat([dec3, conv2], 1), self.upsample(attmap3))
        dec2 = self.dec2(att2)
        att1, attmap1 = self.att1(torch.cat([dec2, conv1], 1), self.upsample(attmap2))
        dec1 = self.dec1(att1)

        # output = self.final(dec1)
        if self.num_classes > 1:
            output = F.log_softmax(self.final(dec1), dim=1)
        else:
            output = self.final(dec1)
        return output
