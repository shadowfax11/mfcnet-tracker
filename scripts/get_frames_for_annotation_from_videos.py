"""
This script is used to extract frames from videos for annotation.
Author: Bhargav Ghanekar
"""
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from os.path import isfile, join
import sys
sys.path.append('.')
import cv2 
import argparse
from pathlib import Path 
from pylab import *
import random 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing videos')
    parser.add_argument('--vid_ext', type=str, required=True, 
                        help='Extension of video files')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save frames')
    parser.add_argument('--num_samples', type=int, default=225,
                        help='Number of frames to sample')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    args = parser.parse_args()
    main_worker(args)

def main_worker(args): 
    # save_dir should of the format Path('path/to/data/video_idx/images/')
    args.save_dir = Path(args.save_dir)         
    os.makedirs(args.save_dir, exist_ok=True)
    
    np.random.seed(args.seed)
    num = 0 
    video_frame_readout_fps = 5 
    
    video_files = [join(args.data_dir, f) \
                   for f in os.listdir(args.data_dir) \
                    if isfile(join(args.data_dir, f)) and f.endswith(args.vid_ext)]
    video_file = random.choice(video_files)
    print('Choosing video file:', video_file)
    vidObj = cv2.VideoCapture(video_file)
    N = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success = 1
    start_saving = False
    while success:
        success, image = vidObj.read()
        if count % video_frame_readout_fps == 0:
            if not start_saving and np.random.randn()>1.0:
                start_saving = True
                print('Started saving at {} FPS from {}secs'.format(video_frame_readout_fps, count/30))
            if start_saving:
                cv2.imwrite(str(args.save_dir / 'frame{:03d}.jpg'.format(num)), image)
                num += 1
            if num == args.num_samples:
                break
        count += 1

if __name__ == '__main__':
    main()
