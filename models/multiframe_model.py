import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .ternausnet import TernausNet11, TernausNet16
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from segmentation_models_pytorch import Segformer
from hrnet import HighResolutionNet

import torch
import torch.nn as nn

class MultiFrameNetBase(nn.Module):
    def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
        super(MultiFrameNetBase, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.with_optflow = with_optflow
        self.with_depth = with_depth

        # self.in_channels = self.num_frames * self.num_classes
        if has_base_perframe_model_trained:
            self.in_channels = self.num_frames * self.num_classes
        else:
            self.in_channels = 1 * self.num_frames * self.num_classes

        if self.with_optflow:
            self.in_channels += 2 * (self.num_frames - 1)

        if self.with_depth:
            self.in_channels += 1 * self.num_frames

    def forward(self, x):
        raise NotImplementedError("This is a base class. Use MultiFrameNetBasic or MultiFrameNetLarge.")

# class MultiFrameNetBasic(MultiFrameNetBase):
#     def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
#         super(MultiFrameNetBasic, self).__init__(num_classes, num_frames, has_base_perframe_model_trained, with_optflow, with_depth)

#         self.multiframe_net = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, stride=1, padding=5, bias=False),
#             nn.BatchNorm2d(self.num_frames * self.num_classes), 
#             nn.ReLU(),
#             nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
#         )

#     def forward(self, x):
#         return self.multiframe_net(x)

class MultiFrameNetBasic(MultiFrameNetBase):
    def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
        super(MultiFrameNetBasic, self).__init__(num_classes, num_frames, has_base_perframe_model_trained, with_optflow, with_depth)
        self.in_channels = num_classes * num_frames
        if with_depth:
            self.in_channels += num_frames
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.with_optflow = with_optflow
        self.with_depth = with_depth
        
        self.multiframe_net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes), 
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # self.multiframe_net = nn.Sequential(
        #     nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, stride=1, padding=5, bias=False),
        #     nn.BatchNorm2d(self.num_frames * self.num_classes), 
        #     nn.ReLU(),
        #     nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        # )

        # Register the mesh grid as a buffer
        self.register_buffer('grid', self._create_mesh_grid())

    def forward(self, x):
        if self.with_optflow:
            x = self.warp_segmentation_and_depth(x)
        return self.multiframe_net(x)

    def warp_segmentation_and_depth(self, x):
        """
        Warp segmentation maps and depth maps according to the optical flow.

        Args:
            x (torch.Tensor): Input tensor of shape (B, NK + 2K - 2 + K, H, W), where the first NK channels are segmentation maps,
                             the next 2K-2 channels are optical flow maps, and the last K channels are depth maps.

        Returns:
            torch.Tensor: Warped segmentation and depth maps of shape (B, NK + K, H, W).
        """
        N = self.num_classes
        K = self.num_frames
        
        # Split the input into segmentation maps, optical flow maps, and depth maps
        segmentation = x[:, :N*K, :, :]
        flow = x[:, N*K:N*K + 2*K - 2, :, :]
        depth = x[:, N*K + 2*K - 2:, :, :] if self.with_depth else None

        # Warp each segmentation map and depth map using the corresponding optical flow
        warped_segmentations = []
        warped_depths = []
        for i in range(1,K):
            flow_i = flow[:, 2*(i-1):2*i, :, :]
            for j in range(N):
                segmentation_ij = segmentation[:, i*N + j:i*N + j + 1, :, :]
                warped_segmentation_ij = self._warp_single_map(segmentation_ij, flow_i)
                warped_segmentations.append(warped_segmentation_ij)

            if self.with_depth:
                depth_i = depth[:, i:i+1, :, :]
                warped_depth_i = self._warp_single_map(depth_i, flow_i)
                warped_depths.append(warped_depth_i)
        
        # Append the first frame segmentation map without warping
        warped_segmentations.insert(0, segmentation[:, 0:N, :, :])
        
        # Append the first frame depth map without warping
        if self.with_depth:
            depth_0 = depth[:, 0:1, :, :]
            warped_depths.insert(0, depth_0)

        # Concatenate warped segmentation maps along the channel dimension
        warped_segmentation = torch.cat(warped_segmentations, dim=1)

        if self.with_depth:
            # Concatenate warped depth maps along the channel dimension
            warped_depth = torch.cat(warped_depths, dim=1)
            return torch.cat((warped_segmentation, warped_depth), dim=1)
        else:
            return warped_segmentation

    def _warp_single_map(self, map, flow):
        """
        Warp a map (segmentation or depth) according to the RAFT optical flow output.

        Args:
            map (torch.Tensor): Map of shape (B, 1, H, W).
            flow (torch.Tensor): Optical flow map of shape (B, 2, H, W), where flow[:, 0, :, :] is the x-component
                                 and flow[:, 1, :, :] is the y-component of the flow.

        Returns:
            torch.Tensor: Warped map of shape (B, 1, H, W).
        """
        _, _, H, W = map.size()

        # Use the precomputed mesh grid (normalized to [-1, 1] for grid_sample)
        grid = self.grid[:, :, :H, :W]  # Shape: (1, 2, H, W)

        # Add the flow to the grid using broadcasting instead of repeating the grid
        flow_x = flow[:, 0, :, :] / ((W - 1) / 2.0)
        flow_y = flow[:, 1, :, :] / ((H - 1) / 2.0)
        flow = torch.stack((flow_x, flow_y), dim=1)
        new_grid = grid + flow  # Shape: (B, 2, H, W)

        # Permute grid to (B, H, W, 2) for grid_sample
        new_grid = new_grid.permute(0, 2, 3, 1)

        # Warp the map using the computed flow grid
        warped_map = F.grid_sample(map, new_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_map

    def _create_mesh_grid(self):
        """
        Create a mesh grid for the image dimensions, normalized to [-1, 1] for use in grid_sample.

        Returns:
            torch.Tensor: Mesh grid of shape (1, 2, H, W).
        """
        H, W = 576, 720  # Default size, will be cropped/resized as needed
        y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_y = 2.0 * y / (H - 1) - 1.0
        grid_x = 2.0 * x / (W - 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # Shape: (2, H, W)
        grid = grid.unsqueeze(0)  # Shape: (1, 2, H, W)
        return grid

class MultiFrameNetLarge(MultiFrameNetBase):
    def __init__(self, num_classes, num_frames, has_base_perframe_model_trained=False, with_optflow=False, with_depth=False):
        super(MultiFrameNetLarge, self).__init__(num_classes, num_frames, has_base_perframe_model_trained, with_optflow, with_depth)

        self.multiframe_net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_frames * self.num_classes, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes), 
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_frames * self.num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.num_frames * self.num_classes),
            nn.ReLU(),
            nn.Conv2d(self.num_frames * self.num_classes, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.multiframe_net(x)

class TernausNetMultiBasic(nn.Module):
    def __init__(self, num_classes, num_frames, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(TernausNetMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        if loadpath is not None:
            self.base_model = TernausNet16(num_classes=self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = True
        else:
            self.base_model = TernausNet16(num_classes=1*self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img).exp()
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output

class TernausNetMultiLarge(nn.Module):
    def __init__(self, num_classes, num_frames, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(TernausNetMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        if loadpath is not None:
            self.base_model = TernausNet16(num_classes=self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = True
        else:
            self.base_model = TernausNet16(num_classes=1*self.num_classes, num_filters=64, pretrained=self.pretrained)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img).exp()
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output

class DeepLabMultiBasic(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(DeepLabMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.deeplabv3_resnet101(pretrained=self.pretrained, progress=True)
        if loadpath is not None: 
            self.base_model.classifier = DeepLabHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = DeepLabHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)['out']
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output

class DeepLabMultiLarge(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(DeepLabMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.deeplabv3_resnet101(pretrained=self.pretrained, progress=True)
        if loadpath is not None: 
            self.base_model.classifier = DeepLabHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = DeepLabHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)['out']
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class SegFormerMultiBasic(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(SegFormerMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = Segformer(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=self.num_classes, activation='logsoftmax')
        if loadpath is not None: 
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class SegFormerMultiLarge(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False):
        super(SegFormerMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = Segformer(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=self.num_classes, activation='logsoftmax')
        if loadpath is not None: 
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)

    def forward(self, x, optflow=None, depth=None): 
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class HRNetMultiBasic(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(HRNetMultiBasic, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = HighResolutionNet(num_classes=self.num_classes)
        if loadpath is not None:
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class HRNetMultiLarge(nn.Module):
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(HRNetMultiLarge, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = HighResolutionNet(num_classes=self.num_classes)
        if loadpath is not None:
            has_base_preframe_model_trained = True
        else:
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained,
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs)
    
    def forward(self, x, optflow=None, depth=None):
        y_output = []
        for x_img in x: 
            y_img = self.base_model(x_img)
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output


class FCNMultiBasic(nn.Module): 
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(FCNMultiBasic, self).__init__()
        self.num_classes = num_classes 
        self.num_frames = num_frames 
        self.pretrained = pretrained 
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.fcn_resnet101(pretrained=self.pretrained, progress=True) 
        if loadpath is not None: 
            self.base_model.classifier = FCNHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = FCNHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetBasic(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs) 
    
    def forward(self, x, optflow=None, depth=None):
        y_output = [] 
        for x_img in x: 
            y_img = self.base_model(x_img)['out'] 
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output

class FCNMultiLarge(nn.Module): 
    def __init__(self, num_classes=2, num_frames=1, pretrained=True, loadpath=None, optflow_inputs=False, depth_inputs=False): 
        super(FCNMultiLarge, self).__init__()
        self.num_classes = num_classes 
        self.num_frames = num_frames 
        self.pretrained = pretrained 
        self.optflow_inputs = optflow_inputs
        self.depth_inputs = depth_inputs
        self.base_model = models.segmentation.fcn_resnet101(pretrained=self.pretrained, progress=True) 
        if loadpath is not None: 
            self.base_model.classifier = FCNHead(2048, self.num_classes)
            has_base_preframe_model_trained = True
        else:
            self.base_model.classifier = FCNHead(2048, 1*self.num_classes)
            has_base_preframe_model_trained = False
        self.multiframe_net = MultiFrameNetLarge(self.num_classes, self.num_frames, has_base_preframe_model_trained, 
                                            with_optflow=self.optflow_inputs, with_depth=self.depth_inputs) 
    
    def forward(self, x, optflow=None, depth=None):
        y_output = [] 
        for x_img in x: 
            y_img = self.base_model(x_img)['out'] 
            y_output.append(y_img)                  # B x 2N_c x H x W
        if optflow is not None: 
            for optflow_img in optflow: 
                y_output.append(optflow_img)        # Add 2*(N_f-1) optical flow images
        if depth is not None:
            for depth_img in depth: 
                y_output.append(depth_img)          # Add N_f depth images
        
        y_output = torch.cat(y_output, dim=1)       # B x H x W
        y_output = self.multiframe_net(y_output)    # B x N_c x H x W
        return y_output
