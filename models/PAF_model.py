"""
Model built on Du et al. 2018 paper
Author: Bhargav Ghanekar
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class SBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(SBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x

class DBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(DBR, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PAF_detection_model(nn.Module):
    def __init__(self, num_joints, num_joint_associations):
        super(PAF_detection_model, self).__init__()
        self.N_joints = num_joints
        self.N_joint_associations = num_joint_associations
        self.CBR0 = CBR(3, 64, kernel_size=3, stride=1, padding=1)
        
        self.SBR1a = SBR(64, 64, kernel_size=2, stride=2, padding=0)
        self.CBR1a = CBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.SBR1b = SBR(64, 64, kernel_size=2, stride=2, padding=0)
        self.CBR1b = CBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.CBR1 = CBR(128, 128, kernel_size=1, stride=1, padding=0)

        self.SBR2a = SBR(128, 128, kernel_size=2, stride=2, padding=0)
        self.CBR2a = CBR(128, 128, kernel_size=3, stride=1, padding=1)
        self.SBR2b = SBR(128, 128, kernel_size=2, stride=2, padding=0)
        self.CBR2b = CBR(128, 128, kernel_size=3, stride=1, padding=1)
        self.CBR2 = CBR(256, 256, kernel_size=1, stride=1, padding=0)

        self.SBR3a = SBR(256, 256, kernel_size=2, stride=2, padding=0)
        self.CBR3a = CBR(256, 256, kernel_size=3, stride=1, padding=1)
        self.SBR3b = SBR(256, 256, kernel_size=2, stride=2, padding=0)
        self.CBR3b = CBR(256, 256, kernel_size=3, stride=1, padding=1)
        self.CBR3 = CBR(512, 512, kernel_size=1, stride=1, padding=0)

        self.SBR4a = SBR(512, 512, kernel_size=2, stride=2, padding=0)
        self.CBR4a = CBR(512, 512, kernel_size=3, stride=1, padding=1)
        self.SBR4b = SBR(512, 512, kernel_size=2, stride=2, padding=0)
        self.CBR4b = CBR(512, 512, kernel_size=3, stride=1, padding=1)
        self.CBR4 = CBR(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.DBR5a = DBR(512, 256, kernel_size=2, stride=2, padding=0)
        self.CBR5a = CBR(256, 256, kernel_size=3, stride=1, padding=1)
        self.DBR5b = DBR(512, 256, kernel_size=2, stride=2, padding=0)
        self.CBR5b = CBR(256, 256, kernel_size=3, stride=1, padding=1)
        self.CBR5 = CBR(512, 512, kernel_size=1, stride=1, padding=0)

        self.DBR6a = DBR(256, 128, kernel_size=2, stride=2, padding=0)
        self.CBR6a = CBR(128, 128, kernel_size=3, stride=1, padding=1)
        self.DBR6b = DBR(256, 128, kernel_size=2, stride=2, padding=0)
        self.CBR6b = CBR(128, 128, kernel_size=3, stride=1, padding=1)
        self.CBR6 = CBR(256, 256, kernel_size=1, stride=1, padding=0)

        self.DBR7a = DBR(128, 64, kernel_size=2, stride=2, padding=0)
        self.CBR7a = CBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.DBR7b = DBR(128, 64, kernel_size=2, stride=2, padding=0)
        self.CBR7b = CBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.CBR7 = CBR(128, 128, kernel_size=1, stride=1, padding=0)

        self.DBR8a = DBR(64, 32, kernel_size=2, stride=2, padding=0)
        self.CBR8a = CBR(32, 32, kernel_size=3, stride=1, padding=1)
        self.DBR8b = DBR(64, 32, kernel_size=2, stride=2, padding=0)
        self.CBR8b = CBR(32, 32, kernel_size=3, stride=1, padding=1)
        self.CBR8 = CBR(64, 64, kernel_size=1, stride=1, padding=0)

        self.CBS9a = nn.Conv2d(64, self.num_joints, kernel_size=1, stride=1, padding=0)
        self.CBS9b = nn.Conv2d(64, self.num_joint_associations, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.CBR0(x)

        xa = self.CBR1a(self.SBR1a(x))
        xb = self.CBR1b(self.SBR1b(x))
        x1 = self.CBR1(torch.cat((xa, xb), 1))
        
        xa = self.CBR2a(self.SBR2a(x1))
        xb = self.CBR2b(self.SBR2b(x1))
        x2 = self.CBR2(torch.cat((xa, xb), 1))

        xa = self.CBR3a(self.SBR3a(x2))
        xb = self.CBR3b(self.SBR3b(x2))
        x3 = self.CBR3(torch.cat((xa, xb), 1))

        xa = self.CBR4a(self.SBR4a(x3))
        xb = self.CBR4b(self.SBR4b(x3))
        x4 = self.CBR4(torch.cat((xa, xb), 1))

        x4a, x4b = torch.chunk(x4, 2, dim=1)
        xa = self.CBR5a(self.DBR5a(x4a))
        xb = self.CBR5b(self.DBR5b(x4b))
        x5 = self.CBR5(torch.cat((xa, xb), 1))
        x5 = x5 + x3

        x5a, x5b = torch.chunk(x5, 2, dim=1)
        xa = self.CBR6a(self.DBR6a(x5a))
        xb = self.CBR6b(self.DBR6b(x5b))
        x6 = self.CBR6(torch.cat((xa, xb), 1))
        x6 = x6 + x2

        x6a, x6b = torch.chunk(x6, 2, dim=1)
        xa = self.CBR7a(self.DBR7a(x6a))
        xb = self.CBR7b(self.DBR7b(x6b))
        x7 = self.CBR7(torch.cat((xa, xb), 1))
        
        x7a, x7b = torch.chunk(x7, 2, dim=1)
        x8a = self.CBR8a(self.DBR8a(x7a))
        x8b = self.CBR8b(self.DBR8b(x7b))

        x9a = self.CBS9a(x8a)
        x9b = self.CBS9b(x8b)

        return x9a, x9b
