"""
Taken from MF-TAPNet official repository.
"""
import torch 
import torch.nn as nn 
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms.functional as tF
import math 
import numpy as np
import cv2 
from pathlib import Path 
import tqdm 
import argparse 

import sys
sys.path.append('.')
from src.dataloader import get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames
from utils.vis_utils import flow_to_arrow, flow_to_color

from torchvision.models.optical_flow import raft_large

def rescale_flow(flow, original_size, scale_factor):
    new_height, new_width = int(original_size[0] * scale_factor), int(original_size[1] * scale_factor)
    flow_rescaled = cv2.resize(flow, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    flow_rescaled[:, :, 0] *= original_size[1] / new_width
    flow_rescaled[:, :, 1] *= original_size[0] / new_height
    return flow_rescaled

def estimate(model, x_prev, x_curr, args):
    '''
    given image pair <x_prev, x_curr> (h, w), estimate the dense optical flow (h, w, 2)
    @param x_prev: first images batch 4d-tensor (b, c=3, h, w)
    @param x_curr: second images batch 4d-tensor (b, c=3, h, w)
    return: optical flow for pairs batch 4d-tensor (b, c=2, h, w)
    '''
    h, w = x_prev.size()[2:]        # original size
    default_h = args.input_height
    default_w = args.input_width
    assert h == args.original_height and w == args.original_width, f"input size should be {args.original_height}x{args.original_width}"
    assert (h / default_h) == (w / default_w), "input size should be the same ratio"
    scale_factor = (default_h / h)

    x_prev = interpolate(input=x_prev, size=(default_h, default_w), mode='bilinear', align_corners=True).cuda()
    x_curr = interpolate(input=x_curr, size=(default_h, default_w), mode='bilinear', align_corners=True).cuda()
    output = model(x_prev.cuda(), x_curr.cuda())
    output = output[-1]
    output = rescale_flow(output.cpu().squeeze().numpy().transpose(1, 2, 0), (h, w), scale_factor=scale_factor)
    return output

class ImagePairsDataset(Dataset):
    """
    ImagePairsDataset: return image pairs (consecutive frames) for optical flow estimation
    """
    def __init__(self, filenames, mode):
        super(ImagePairsDataset, self).__init__()
        self.filenames = filenames
        self.mode = mode

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # an exception for each video:
        # for the first frame in each video, the optical flow should be 0
        # optflow[0] = flow<0, 0> = 0, optflow[k] = flow<k-1, k> (k > 0)
        NUM_FRAMES_VIDEO = 225 if self.mode=='train' else 75
        first_idx = idx if idx % NUM_FRAMES_VIDEO == 0 else idx - 1
        next_idx = idx
        file1, file2 = self.filenames[first_idx], self.filenames[next_idx]
        # according to the UnFlow implementation, inputs are in normalized BGR space
        first = cv2.imread(str(file1))
        first = cv2.resize(first, (640, 480), interpolation=cv2.INTER_LINEAR)
        second = cv2.imread(str(file2))
        second = cv2.resize(second, (640, 480), interpolation=cv2.INTER_LINEAR)
        # img_to_tensor will reshape into (c, h, w) and scaled to [0., 1.]
        first = torch.from_numpy(first).float().permute(2, 0, 1) / 255.
        first = tF.normalize(first, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        second = torch.from_numpy(second).float().permute(2, 0, 1) / 255.
        second = tF.normalize(second, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return str(file2), first, second

def main(args):
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    torch.cuda.device(0) # use one GPU

    # load model
    model = raft_large(pretrained=True, progress=False).cuda()
    model = model.eval()

    # dataloader
    args.fold_index = -1
    args.data_dir = Path(args.data_dir)
    if args.dataset == 'MICCAI2017':    
        filenames, _ = get_MICCAI2017_dataset_filenames(args)
    elif args.dataset == 'JIGSAWS':
        filenames, _ = get_JIGSAWS_dataset_filenames(args)
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    loader = DataLoader(dataset=ImagePairsDataset(filenames, mode=args.mode),
                        shuffle=False, # no need to shuffle
                        num_workers=0, # pretrained model not support parallel
                        batch_size=1,
                        pin_memory=True)
    
    # progress bar
    tq = tqdm.tqdm(total=len(loader.dataset),
        desc='estimate optical flow for image pairs')
    
    for i, (filename, x_prev, x_curr) in enumerate(loader):
        outputs = estimate(model, x_prev, x_curr, args)
        
        # for filename, output in enumerate(fname_curr, outputs):
        flow_uv = outputs #.numpy().transpose(1, 2, 0) # (h, w, c)
        filename = Path(filename[0])

        # save optical flow in instruments_dataset_X/optflows/filename.flo
        video_dir = filename.parent.parent
        # optflow_dir = video_dir / args.optflow_dir
        # optflow_dir.mkdir(exist_ok=True, parents=True)
        # optfilename = optflow_dir / (filename.stem + ".flo")
        # objectOutput = open(str(optfilename), 'wb')
        # # np.array([80], np.uint8).tofile(objectOutput)
        # np.array([output.size(2), output.size(1)], np.int32).tofile(objectOutput)
        # # store in (h, w, c)
        # np.array(flow_uv, np.float32).tofile(objectOutput)
        # objectOutput.close()

        if (args.visualize):
            # save optical flow visualization in color model
            # in instruments_dataset_X/optflows/filename
            optflow_vis_color_dir = video_dir / args.optflow_vis_color_dir
            optflow_vis_color_dir.mkdir(exist_ok=True, parents=True)
            flow_color = flow_to_color(flow_uv, convert_to_bgr=False)
            cv2.imwrite(str(optflow_vis_color_dir / (filename.name)), flow_color)

            # save optical flow visualization in arrows
            # in instruments_dataset_X/optflows/filename
            # optflow_vis_arrow_dir = video_dir / args.optflow_vis_arrow_dir
            # optflow_vis_arrow_dir.mkdir(exist_ok=True, parents=True)
            # flow_arrow = flow_to_arrow(flow_uv)
            # cv2.imwrite(str(optflow_vis_arrow_dir / (filename.name)), flow_arrow)
                
        tq.update(1)

    tq.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use RAFT to calculate optical flow for train data')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='store the visualization of the optical flow')
    parser.add_argument('--data_dir', type=str, default='/home/bg40/surgical_video_datasets/miccai2017/',
                        help='data directory.')
    parser.add_argument('--dataset', type=str, default='MICCAI2017', choices=['MICCAI2017', 'JIGSAWS'],
                        help='dataset name. e.g. MICCAI2017, JIGSAWS')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'],)
    parser.add_argument('--original_height', type=int, default=1024)
    parser.add_argument('--original_width', type=int, default=1280)
    parser.add_argument('--input_height', type=int, default=1024, 
                        help='input height for optical flow model')
    parser.add_argument('--input_width', type=int, default=1280,
                        help='input width for optical flow model')
    parser.add_argument('--optflow_dir', type=str, default='optflows',
                        help='optical flow file save dir. e.g. .../instrument_dataset_X/optflow_dir/*.flo')
    parser.add_argument('--optflow_vis_color_dir', type=str, default='optflows_vis_color',
                        help='visualization of optical flow in color model save dir. e.g. .../instrument_dataset_X/optflow_vis_color_dir/*.png')
    parser.add_argument('--optflow_vis_arrow_dir', type=str, default='optflows_vis_arrow',
                        help='visualization of optical flow in arrows save dir. e.g. .../instrument_dataset_X/optflow_vis_arrow_dir/*.png')
    args = parser.parse_args()
    main(args)

