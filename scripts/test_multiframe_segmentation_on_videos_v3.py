"""
Script for testing the multiframe segmentation model on videos. 
Can be used for tool-tip/pose segmentation.
Author: Bhargav Ghanekar
"""

import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys; sys.path.append('.'); sys.path.append('./models/')
import logging, json, argparse 
from pathlib import Path
import cv2 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms import functional as tF

import matplotlib.pyplot as plt
from models import get_multiframe_segmentation_model as get_model
from utils.model_utils import load_model_weights
from utils.vis_utils import mask_overlay
from utils.localization_utils_v2 import create_circular_mask, determine_local_maxima_and_estimate_centroids
import tqdm.auto as tqdm

# auxiliary functions
def compute_distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def refine_tip_segmentation(mask, args): 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # Find contours in the binary image
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]                  # Sort contours by area (largest first)
    blob_selector = np.zeros_like(mask)                                                 # Create a mask to hold the two largest blobs
    for c in contours:                                                                  # Draw the two largest contours on the mask
        area = cv2.contourArea(c)
        if area < args.area_threshold:
            continue
        cv2.drawContours(blob_selector, [c], 0, (255), thickness=cv2.FILLED)
    mask_refined = cv2.bitwise_and(mask, mask, mask=blob_selector)                      # Apply the mask to the original binary image
    return mask_refined


def calc_base_centroid(mask, args):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    cX = []
    cY = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < args.area_threshold:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX.append(int(M["m10"] / M["m00"]))
        cY.append(int(M["m01"] / M["m00"]))
    return cX, cY

def main():
    parser = argparse.ArgumentParser(description='Test Multiframe Segmentation Model on Videos')
    parser.add_argument('--videos_dir', type=str, required=True, 
                        help='Path to the directory containing videos')
    parser.add_argument('--depth_videos_dir', type=str, default=None,
                        help='Path to the directory containing depth videos')
    parser.add_argument('--expt_savedir', type=str, required=True, 
                        help='Path to the directory where the output will be saved')
    parser.add_argument('--expt_name', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['TernausNetMulti-Basic', 'TernausNetMulti-Large', 'DeepLabMulti-Basic', 'DeepLabMulti-Large', 'FCNMulti-Basic', 'FCNMulti-Large'],
                        help='Type of model to use')
    parser.add_argument('--num_input_frames', type=int, default=3, 
                        help='Number of input frames to the model. Default: 3')
    parser.add_argument('--load_wts_model', type=str, required=True,
                        help='Path to the model weights')
    parser.add_argument('--num_videos', type=int, default=-1,
                        help='Number of videos to process. Default: -1, indicates to process all videos')
    parser.add_argument('--input_width', type=int, default=640)
    parser.add_argument('--input_height', type=int, default=480)
    parser.add_argument('--score_detection_threshold', type=float, default=0, 
                        help='Threshold for detection score')
    parser.add_argument('--area_threshold', type=int, default=10, 
                        help='Threshold for area of detected contour/tooltip')
    parser.add_argument('--dist_threshold', type=int, default=40, 
                        help='Threshold for distance between detected toolbase and tooltip')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False,
                        help='Add optical flow inputs to the model. Default: False')
    parser.add_argument('--add_depth_inputs', type=bool, default=False,
                        help='Add depth inputs to the model. Default: False')
    args = parser.parse_args()
    main_worker(args)

def compute_centroids_and_store(type, mask_array, output, centroid_locations, count, args, disp_image, prev_left_pose_detected_tips, cX_prev_left, cY_prev_left):
    if type=='left': 
        idxt1 = 0
        idxt2 = 1
        idxt3 = 2
        idxt4 = 3
        idxb1 = 8
        idxb2 = 9
        colors = (255,255,255)
        left_base = 255*(mask_array==3).astype(np.uint8)
        left_tip = 255*(mask_array==4).astype(np.uint8)
        left_tip_heatmap = output[0,4,:,:].cpu().numpy()
    elif type=='right':
        idxt1 = 4
        idxt2 = 5
        idxt3 = 6
        idxt4 = 7
        idxb1 = 10
        idxb2 = 11
        colors = (0,0,0)
        left_base = 255*(mask_array==1).astype(np.uint8)
        left_tip = 255*(mask_array==2).astype(np.uint8)
        left_tip_heatmap = output[0,2,:,:].cpu().numpy()
    
    fmask = create_circular_mask(10,10).astype(np.float64)
    left_tip_heatmap[left_tip==0] = 0
    iX_left, iY_left = calc_base_centroid(left_base, args)
    if len(iX_left) == 0 and len(iY_left) == 0:
        prev_left_pose_detected_tips = 0 
    elif len(iX_left) == 1 and len(iY_left) == 1:
        centroid_locations[count, idxb1] = iX_left[0]
        centroid_locations[count, idxb2] = iY_left[0]
        left_tip = refine_tip_segmentation(left_tip, args)
        cX_left, cY_left = determine_local_maxima_and_estimate_centroids(left_tip_heatmap, left_tip>0, fmask)
        if len(cX_left)==0 and len(cY_left)==0:
            prev_left_pose_detected_tips = 0
        elif len(cX_left)==1 and len(cY_left)==1:
            d01 = compute_distance(iX_left[0], iY_left[0], cX_left[0], cY_left[0])
            if d01 < args.dist_threshold:
                prev_left_pose_detected_tips = 1
                centroid_locations[count, idxt1] = cX_left[0]
                centroid_locations[count, idxt2] = cY_left[0]
                centroid_locations[count, idxt3] = cX_left[0]
                centroid_locations[count, idxt4] = cY_left[0]
                cv2.circle(disp_image, (cX_left[0], cY_left[0]), 4, colors, -1)
            else:
                prev_left_pose_detected_tips = 0
        elif len(cX_left)==2 and len(cY_left)==2:
            d01 = compute_distance(iX_left[0], iY_left[0], cX_left[0], cY_left[0])
            d02 = compute_distance(iX_left[0], iY_left[0], cX_left[1], cY_left[1])
            if d01 < args.dist_threshold and d02 < args.dist_threshold:
                prev_left_pose_detected_tips = 2
                d11 = compute_distance(cX_left[0], cY_left[0], cX_prev_left[0], cY_prev_left[0])
                d12 = compute_distance(cX_left[0], cY_left[0], cX_prev_left[1], cY_prev_left[1])
                d21 = compute_distance(cX_left[1], cY_left[1], cX_prev_left[0], cY_prev_left[0])
                d22 = compute_distance(cX_left[1], cY_left[1], cX_prev_left[1], cY_prev_left[1])
                if d11+d22 < d12+d21:
                    centroid_locations[count, idxt1] = cX_left[0]
                    centroid_locations[count, idxt2] = cY_left[0]
                    centroid_locations[count, idxt3] = cX_left[1]
                    centroid_locations[count, idxt4] = cY_left[1]
                else:
                    centroid_locations[count, idxt1] = cX_left[1]
                    centroid_locations[count, idxt2] = cY_left[1]
                    centroid_locations[count, idxt3] = cX_left[0]
                    centroid_locations[count, idxt4] = cY_left[0]
                cv2.circle(disp_image, (cX_left[0], cY_left[0]), 4, colors, -1)
                cv2.circle(disp_image, (cX_left[1], cY_left[1]), 4, colors, -1)
            else: 
                if d01 < args.dist_threshold:
                    prev_left_pose_detected_tips = 1
                    centroid_locations[count, idxt1] = cX_left[0]
                    centroid_locations[count, idxt2] = cY_left[0]
                    centroid_locations[count, idxt3] = cX_left[0]
                    centroid_locations[count, idxt4] = cY_left[0]
                    cv2.circle(disp_image, (cX_left[0], cY_left[0]), 4, colors, -1)
                elif d02 < args.dist_threshold:
                    prev_left_pose_detected_tips = 1
                    centroid_locations[count, idxt1] = cX_left[1]
                    centroid_locations[count, idxt2] = cY_left[1]
                    centroid_locations[count, idxt3] = cX_left[1]
                    centroid_locations[count, idxt4] = cY_left[1]
                    cv2.circle(disp_image, (cX_left[1], cY_left[1]), 4, colors, -1)
                else: 
                    prev_left_pose_detected_tips = 0
        else: 
            raise ValueError(f"Unexpected number of detected tips: {len(cX_left)}")

        if type=='left':
            cX_prev_left = centroid_locations[count, 0:4:2]
            cY_prev_left = centroid_locations[count, 1:4:2]
        if type=='right':
            cX_prev_left = centroid_locations[count, 4:8:2]
            cY_prev_left = centroid_locations[count, 5:8:2]
        # iX_prev_left = centroid_locations[count, 8]
        # iY_prev_left = centroid_locations[count, 9]
        cv2.circle(disp_image, (iX_left[0], iY_left[0]), 2, colors, -1)
    else:
        raise ValueError(f"Unexpected number of detected bases: {len(iX_left)}")
    return centroid_locations, prev_left_pose_detected_tips, cX_prev_left, cY_prev_left, disp_image



def track_on_video(video_path, depth_video_path, model, args, logger, optflow_model=None):
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    model.eval()
    output_fps = 30 
    vidObj = cv2.VideoCapture(video_path)
    fname, _ = os.path.splitext(video_path.replace('\\', '/').split('/')[-1])
    N = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of frames in video {fname}: {N}")
    if args.add_depth_inputs:
        depth_vidObj = cv2.VideoCapture(depth_video_path)
        fname_depth, _ = os.path.splitext(depth_video_path.replace('\\', '/').split('/')[-1])
        print(f"Number of frames in depth video {fname_depth}: {int(depth_vidObj.get(cv2.CAP_PROP_FRAME_COUNT))}")
        assert N == int(depth_vidObj.get(cv2.CAP_PROP_FRAME_COUNT)), "Number of frames in RGB and depth videos should be the same"
    count = 0 

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(args.output_dir, f'{fname}_tracked.mp4'), fourcc, output_fps, (args.input_width, args.input_height))
    
    # Initialize variables for tracking
    centroid_locations = np.zeros((N, 12))
    centroid_locations[:,:] = np.nan
    prev_left_pose_detected_tips = 0 
    prev_right_pose_detected_tips = 0
    cX_prev_left = np.zeros(2)
    cY_prev_left = np.zeros(2)
    cX_prev_right = np.zeros(2)
    cY_prev_right = np.zeros(2)
    image_queue = []
    depth_image_queue = []
    with torch.no_grad():
        tq = tqdm.tqdm(total=int(N/30))
        tq.set_description("Progress")
        while True: 
            ret, frame = vidObj.read()
            if not ret:
                break 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_queue.append(frame)
            if args.add_depth_inputs:
                ret_depth, frame_depth = depth_vidObj.read()
                if not ret_depth:
                    print(f"Depth video ended before RGB video at frame {count}")
                    break
                frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)
                depth_image_queue.append(frame_depth)
            if len(image_queue) > args.num_input_frames:
                image_queue.pop(0)
            if len(image_queue) == args.num_input_frames:
                input = []
                input_depth = []
                for i in range(args.num_input_frames-1, -1, -1): # read in reverse order
                    img = cv2.resize(image_queue[i], (args.input_width, args.input_height))
                    img = tF.to_tensor(img.astype(np.float32)/255.0).unsqueeze(0)
                    img = tF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    input.append(img)
                    if args.add_depth_inputs:
                        depth_img = cv2.resize(depth_image_queue[i], (args.input_width, args.input_height))
                        depth_img = tF.to_tensor(depth_img.astype(np.float32)/255.0).unsqueeze(0)
                        input_depth.append(depth_img)
                if torch.cuda.is_available():
                    input = [input[i].cuda() for i in range(len(input))]
                    if args.add_depth_inputs:
                        input_depth = [input_depth[i].cuda() for i in range(len(input_depth))]
                if args.add_optflow_inputs:
                    optflow = [] 
                    for i in range(1,len(input)):
                        frame_prev = F.interpolate(input[0], scale_factor=0.5, mode='nearest')
                        frame_curr = F.interpolate(input[i], scale_factor=0.5, mode='nearest')
                        flow = optflow_model(frame_prev, frame_curr)[-1]
                        flow = F.interpolate(flow/0.5, size=(input[0].size(2), input[0].size(3)), mode='bilinear', align_corners=True)
                        optflow.append(flow)
                    if args.add_depth_inputs:
                        output = model(input, optflow=optflow, depth=input_depth)
                    else: 
                        output = model(input, optflow=optflow)
                else:
                    if args.add_depth_inputs:
                        output = model(input, depth=input_depth)
                    else:
                        output = model(input)
                output = torch.exp(F.log_softmax(output, dim=1))
                if args.score_detection_threshold >0:
                    output_classes = np.zeros((args.input_height, args.input_width))
                    output_classes[np.where(output[0,1,:,:].cpu().numpy() > args.score_detection_threshold)] = 1
                    output_classes[np.where(output[0,2,:,:].cpu().numpy() > args.score_detection_threshold)] = 2
                    output_classes[np.where(output[0,3,:,:].cpu().numpy() > args.score_detection_threshold)] = 3
                    output_classes[np.where(output[0,4,:,:].cpu().numpy() > args.score_detection_threshold)] = 4
                else: 
                    output_classes = output.data.cpu().numpy().argmax(axis=1).squeeze()
                
                mask_array = output_classes
                disp_image = cv2.resize(image_queue[-1], (args.input_width, args.input_height))
                disp_image = mask_overlay(disp_image, (mask_array==1).astype(np.uint8), color=(255,1,0))
                disp_image = mask_overlay(disp_image, (mask_array==2).astype(np.uint8), color=(255,255,1))
                disp_image = mask_overlay(disp_image, (mask_array==3).astype(np.uint8), color=(0,1,255))
                disp_image = mask_overlay(disp_image, (mask_array==4).astype(np.uint8), color=(0,255,255))
                
                # get centroid of tooltip and toolbase for left instrument
                centroid_locations, prev_left_pose_detected_tips, cX_prev_left, cX_prev_left, disp_image = compute_centroids_and_store('left', mask_array, output, centroid_locations, count,
                                                                                                                                         args, disp_image, prev_left_pose_detected_tips, 
                                                                                                                                         cX_prev_left, cY_prev_left)
                # get centroid of tooltip and toolbase for right instrument
                centroid_locations, prev_right_pose_detected_tips, cX_prev_right, cY_prev_right, disp_image = compute_centroids_and_store('right', mask_array, output, centroid_locations, count, 
                                                                                                                                          args, disp_image, prev_right_pose_detected_tips,
                                                                                                                                          cX_prev_right, cY_prev_right)
                video.write(cv2.cvtColor(disp_image, cv2.COLOR_RGB2BGR))
            count += 1
            if count % 30 == 0:
                tq.update(1)
        tq.close()
        cv2.destroyAllWindows()
        video.release()
    np.savetxt(os.path.join(args.output_dir, f'{fname}_tracked.csv'), centroid_locations, delimiter=',')
    logger.info(f"Saved tracked results to {os.path.join(args.output_dir, f'{fname}_tracked.csv')}")
    logger.info("Number of missing centroids for left: {}, for right {}".format(np.count_nonzero(np.isnan(centroid_locations[:,0]))/N, np.count_nonzero(np.isnan(centroid_locations[:,4]))/N))
    return

def main_worker(args):
    # setup 
    args.mode = 'testing'
    args.load_wts_base_model = 'test'
    args.videos_dir = Path(args.videos_dir)
    args.output_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'video_tracking_results'))
    for dir in [args.output_dir]:
        print(f"Creating {dir.resolve()} if non-existent")
        dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(os.path.join(args.log_dir, "log.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # set up optical flow model if needed
    if args.add_optflow_inputs:
        from torchvision.models.optical_flow import raft_large
        optflow_model = raft_large(pretrained=True, progress=False).cuda()
        if torch.cuda.is_available():
            optflow_model = optflow_model.cuda()
        else: 
            raise SystemError('GPU device not found! Not configured to train/test.')
        optflow_model.eval()
        logger.info("RAFT optical flow model loaded")
    else: 
        optflow_model = None
    
    # set up model 
    args.num_classes = 5
    args.pretrained = True
    model = get_model(args)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    else: 
        raise SystemError('GPU device not found! Not configured to train/test.')
    
    # load pre-trained weights
    model, _, load_flag = load_model_weights(model, args.load_wts_model, args.model_type)
    if load_flag:
        logger.info(f"Loaded model weights from {args.load_wts_model}")
    else:
        raise ValueError(f"Failed to load model weights from {args.load_wts_model}")
    model.eval()
    
    # get video files
    def list_video_files(directory):
        lst = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    lst.append(os.path.join(root, file))
        return sorted(lst)
    video_files = list_video_files(args.videos_dir)
    video_files = video_files[::1]                                                  # choosing every alternate video
    logger.info(f"Found {len(video_files)} video files in {args.videos_dir}")
    if args.num_videos > 0:
        video_files = video_files[:args.num_videos]
    logger.info(f"Processing {len(video_files)} video files")

    if args.add_depth_inputs:
        depth_video_files = list_video_files(args.depth_videos_dir)
        depth_video_files = depth_video_files[::1]                                      # choosing every alternate video
        assert len(video_files) == len(depth_video_files), "Number of RGB and depth videos should be the same, {} RGB videos and {} depth videos found".format(len(video_files), len(depth_video_files))
    else: 
        depth_video_files = [None]*len(video_files)
    
    # process each video
    idx = 0
    for video_file in video_files:
        logger.info(f"Processing video: {video_file}")
        track_on_video(video_file, depth_video_files[idx], model, args, logger, optflow_model)
        idx += 1

if __name__ == '__main__':
    main()
