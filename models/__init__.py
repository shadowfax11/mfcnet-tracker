import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from segmentation_models_pytorch import Segformer
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision import models
from .ternausnet import TernausNet11, TernausNet16
from .tap_model import TAPNet11, TAPNet16
from .multiframe_model import TernausNetMultiBasic, TernausNetMultiLarge, DeepLabMultiBasic, DeepLabMultiLarge, \
    FCNMultiBasic, FCNMultiLarge, SegFormerMultiBasic, SegFormerMultiLarge, HRNetMultiBasic, HRNetMultiLarge
from .hrnet import HighResolutionNet
from .bn_helper import BatchNorm2d

class IdentityModel(nn.Module):
    def __init__(self, args):
        super(IdentityModel, self).__init__()
    def forward(self, x):
        return x

def get_tooltip_segmentation_model(args):
    if args.model_type == 'TernausNet11':
        model = TernausNet11(num_classes=args.num_classes, num_filters=64, pretrained=args.pretrained)
    elif args.model_type == 'TernausNet16':
        model = TernausNet16(num_classes=args.num_classes, num_filters=64, pretrained=args.pretrained)
    elif args.model_type == 'TAPNet11':
        model = TAPNet11(in_channels=3, num_classes=args.num_classes, pretrained=args.pretrained)
    elif args.model_type == 'TAPNet16':
        model = TAPNet16(in_channels=3, num_classes=args.num_classes, pretrained=args.pretrained)
    elif args.model_type == 'DeepLab_v3':
        model = models.segmentation.deeplabv3_resnet101(pretrained=args.pretrained, progress=True)
        model.classifier = DeepLabHead(2048, args.num_classes)
    elif args.model_type == 'FCN':
        model = models.segmentation.fcn_resnet101(pretrained=args.pretrained, progress=True)
        model.classifier = FCNHead(2048, args.num_classes)
    elif args.model_type == 'HRNet': 
        model = HighResolutionNet()
        model.load_state_dict(torch.load('models/hrnet_cs_8090_torch11.pth'))
        last_inp_channels = model.last_layer[0].in_channels  # Input channels from the penultimate layer
        # Define a new final layer with updated output classes
        model.last_layer = nn.Sequential(nn.Conv2d(in_channels=last_inp_channels,out_channels=last_inp_channels,kernel_size=1,stride=1,padding=0,),
                            BatchNorm2d(last_inp_channels, momentum=0.1),  # Assuming BN_MOMENTUM=0.1
                            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels,out_channels=args.num_classes,kernel_size=1,stride=1,padding=0,))
    elif args.model_type == 'SegFormer':
        # model = FPN(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=args.num_classes, activation='logsoftmax')
        model = Segformer(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=args.num_classes, activation='logsoftmax')
    else:
        raise ValueError(f"Model type {args.model_type} not recognized")
    return model

def get_multiframe_segmentation_model(args):
    if args.model_type == 'TernausNetMulti-Basic':
        model = TernausNetMultiBasic(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'TernausNetMulti-Large':
        model = TernausNetMultiLarge(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'DeepLabMulti-Basic':
        model = DeepLabMultiBasic(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'DeepLabMulti-Large':
        model = DeepLabMultiLarge(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'FCNMulti-Basic': 
        model = FCNMultiBasic(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'FCNMulti-Large':    
        model = FCNMultiLarge(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'SegFormerMulti-Basic':
        model = SegFormerMultiBasic(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'SegFormerMulti-Large':
        model = SegFormerMultiLarge(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,  
                         optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'HRNetMulti-Basic':
        model = HRNetMultiBasic(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,
                                optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    elif args.model_type == 'HRNetMulti-Large':
        model = HRNetMultiLarge(num_classes=args.num_classes, num_frames=args.num_input_frames, pretrained=args.pretrained, loadpath=args.load_wts_base_model,
                                optflow_inputs=args.add_optflow_inputs, depth_inputs=args.add_depth_inputs)
    else:
        raise ValueError(f"Model type {args.model_type} not recognized")
    return model
