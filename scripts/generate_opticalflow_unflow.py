"""
Taken from MF-TAPNet official repository.
"""
import torch 
import torch.nn as nn 
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader 
from albumentations.pytorch.functional import img_to_tensor
import math 
import numpy as np
import cv2 
from pathlib import Path 
import tqdm 
import argparse 

import sys
sys.path.append('.')
from src.dataloader import get_MICCAI2017_dataset_filenames
from utils.vis_utils import flow_to_arrow, flow_to_color

from models.unflow_model import UnFlow

'''
parts of this implementation are borrowed from pytorch UnFlow:
ref: https://github.com/sniklaus/pytorch-unflow
'''
def estimate(model, x_prev, x_curr):
    '''
    given image pair <x_prev, x_curr> (h, w), estimate the dense optical flow (h, w, 2)
    @param x_prev: first images batch 4d-tensor (b, c=3, h, w)
    @param x_curr: second images batch 4d-tensor (b, c=3, h, w)
    return: optical flow for pairs batch 4d-tensor (b, c=2, h, w)
    '''
    assert x_prev.shape == x_curr.shape
    h, w = x_prev.size()[2:]
    # the default input for UnFlow pretrained model is (h * w) = (384 * 1280)
    # for custom input size, you can resize the input to (384 * 1280) and then resize back
    # just input the original shape is OK, but not guaranteed for correctness
    # comment the following lines for unresize
    default_h = 384
    default_w = 1280
    x_prev = interpolate(input=x_prev, size=(default_h, default_w), mode='bilinear', align_corners=True).cuda()
    x_curr = interpolate(input=x_curr, size=(default_h, default_w), mode='bilinear', align_corners=True).cuda()
    assert x_prev.shape[2] == 384
    assert x_curr.shape[3] == 1280
    output = model(x_prev.cuda(), x_curr.cuda())
    # resize back to input shape
    output = interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)
    return output.cpu()

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
        second = cv2.imread(str(file2))
        # img_to_tensor will reshape into (c, h, w) and scaled to [0., 1.]
        return str(file2), img_to_tensor(first), img_to_tensor(second)

def main(args):
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    torch.cuda.device(0) # use one GPU

    # load model
    model = UnFlow().cuda().eval()
    model_path = args.pretrained_opticalflow_model_dir
    model.load_state_dict(torch.load(model_path))

    # dataloader
    args.fold_index = -1
    args.data_dir = Path(args.data_dir)
    filenames, _ = get_MICCAI2017_dataset_filenames(args)
    loader = DataLoader(dataset=ImagePairsDataset(filenames, mode=args.mode),
                        shuffle=False, # no need to shuffle
                        num_workers=0, # pretrained model not support parallel
                        batch_size=1,
                        pin_memory=True)
    
    # progress bar
    tq = tqdm.tqdm(total=len(loader.dataset),
        desc='estimate optical flow for image pairs')
    
    for i, (fname_curr, x_prev, x_curr) in enumerate(loader):
        outputs = estimate(model, x_prev, x_curr)

        for filename, output in zip(fname_curr, outputs):
            flow_uv = output.numpy().transpose(1, 2, 0) # (h, w, c)
            filename = Path(filename)

            # save optical flow in instruments_dataset_X/optflows/filename.flo
            video_dir = filename.parent.parent
            optflow_dir = video_dir / args.optflow_dir
            optflow_dir.mkdir(exist_ok=True, parents=True)
            optfilename = optflow_dir / (filename.stem + ".flo")
            objectOutput = open(str(optfilename), 'wb')
            np.array([80, 73, 69, 72], np.uint8).tofile(objectOutput)
            np.array([output.size(2), output.size(1)], np.int32).tofile(objectOutput)
            # store in (h, w, c)
            np.array(flow_uv, np.float32).tofile(objectOutput)
            objectOutput.close()

            if (args.visualize):
                # save optical flow visualization in color model
                # in instruments_dataset_X/optflows/filename
                optflow_vis_color_dir = video_dir / args.optflow_vis_color_dir
                optflow_vis_color_dir.mkdir(exist_ok=True, parents=True)
                flow_color = flow_to_color(flow_uv, convert_to_bgr=False)
                cv2.imwrite(str(optflow_vis_color_dir / (filename.name)), flow_color)

                # save optical flow visualization in arrows
                # in instruments_dataset_X/optflows/filename
                optflow_vis_arrow_dir = video_dir / args.optflow_vis_arrow_dir
                optflow_vis_arrow_dir.mkdir(exist_ok=True, parents=True)
                flow_arrow = flow_to_arrow(flow_uv)
                cv2.imwrite(str(optflow_vis_arrow_dir / (filename.name)), flow_arrow)
                
        tq.update(1)

    tq.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use UnFlow to calculate optical flow for train data')
    parser.add_argument('--pretrained_opticalflow_model_dir', type=str, default='./assets/unflownetwork-css.pytorch',
                        help='directory of UnFlow pretrained model.')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='store the visualization of the optical flow')
    parser.add_argument('--data_dir', type=str, default='/home/bg40/surgical_video_datasets/miccai2017/',
                        help='data directory.')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'],)
    parser.add_argument('--optflow_dir', type=str, default='optflows',
                        help='optical flow file save dir. e.g. .../instrument_dataset_X/optflow_dir/*.flo')
    parser.add_argument('--optflow_vis_color_dir', type=str, default='optflows_vis_color',
                        help='visualization of optical flow in color model save dir. e.g. .../instrument_dataset_X/optflow_vis_color_dir/*.png')
    parser.add_argument('--optflow_vis_arrow_dir', type=str, default='optflows_vis_arrow',
                        help='visualization of optical flow in arrows save dir. e.g. .../instrument_dataset_X/optflow_vis_arrow_dir/*.png')
    args = parser.parse_args()
    main(args)

