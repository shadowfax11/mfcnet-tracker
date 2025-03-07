"""
Script for running inference for various toolpose segmentation models
Author: Bhargav Ghanekar
"""

import os 
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('.')
sys.path.append('./models/')
import logging
import json
from pathlib import Path

import configargparse
from configs.config_toolposeseg import test_config_parser as config_parser

import logging
import time 
import math 
import cv2 
import random 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms

import matplotlib.pyplot as plt
from src.dataloader import get_data_loader
from src.metrics import get_metrics
from models import get_tooltip_segmentation_model as get_model 
from utils.dataloader_utils import load_image, load_mask, load_optflow_map, load_attmap
from utils.dataloader_utils import get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames, get_MICCAI2015_dataset_filenames
from utils.log_utils import AverageMeter, ProgressMeter, init_logging
from utils.model_utils import load_model_weights
from utils.train_utils import add_metrics_meters
from utils.vis_utils import mask_overlay, draw_plus
from utils.localization_utils_v2 import centroid_error

def main(): 
    parser = configargparse.ArgumentParser() 
    parser = config_parser(parser) 
    args = parser.parse_args() 
    main_worker(args) 

def save_attention_maps(model, args):
    model.eval()
    if args.dataset=='MICCAI2017':
        file_names, _ = get_MICCAI2017_dataset_filenames(args)
    elif args.dataset=='JIGSAWS':
        file_names, _ = get_JIGSAWS_dataset_filenames(args)
    elif args.dataset=='MICCAI2015':
        file_names, _ = get_MICCAI2015_dataset_filenames(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    with torch.no_grad():
        for idx, file_name in enumerate(file_names):
            input = torch.from_numpy(load_image(file_name).transpose(2,0,1))
            input = input.type(torch.float32)/255.
            input = transforms.Resize((args.input_height, args.input_width))(input)
            input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input)
            input = input.unsqueeze(0)
            input = input.cuda(non_blocking=True) if torch.cuda.is_available() else input
            if idx%args.num_frames_per_video==0:
                idx_prev = idx
            else: 
                idx_prev = idx-1
            attmap = load_attmap(file_names, idx, args.num_frames_per_video)
            attmap = torch.from_numpy(attmap).unsqueeze(0).unsqueeze(0)
            attmap = transforms.Resize((args.input_height, args.input_width))(attmap)
            attmap = attmap.cuda(non_blocking=True) if torch.cuda.is_available() else attmap
            output = model(input, attmap)
            output = torch.exp(output) 
            output = torch.sum(output[:,1:,:,:], dim=1, keepdim=False)
            for i in range(input.size(0)):
                cv2.imwrite(str(file_name).replace('images', 'attmaps').replace('jpg', 'png'), (255*output[i].detach().cpu().squeeze().numpy()).astype(np.uint8))
    return

def test(test_dataloader, model, args, test_file_names, logger, writer=None):
    logger.info(f"Testing/Inference")
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f')
    progress_meter_list = [batch_time, data_time]
    progress_meter_list = add_metrics_meters(progress_meter_list, args.metric_fns, args.num_classes)
    progress = ProgressMeter(len(test_dataloader), progress_meter_list)
    model.eval()
    data_time_start = time.time()
    step = 0 
    centroid_pred_err_rt = []
    centroid_pred_err_rb = []
    centroid_pred_err_lt = []
    centroid_pred_err_lb = []
    centroid_pres_err_rt = []
    centroid_pres_err_rb = []
    centroid_pres_err_lt = []
    centroid_pres_err_lb = []
    with torch.no_grad(): 
        for inputs, targets in test_dataloader: 
            data_time.update(time.time() - data_time_start)
            batch_time_start = time.time()
            inputs = inputs.cuda(non_blocking=True) if torch.cuda.is_available() else inputs
            targets = targets.type(torch.LongTensor).cuda(non_blocking=True) if torch.cuda.is_available() else targets.type(torch.LongTensor)
            targets = targets.squeeze(1)
            if 'TernausNet' in args.model_type or 'SegFormer' in args.model_type: 
                outputs = model(inputs)
                # get centroid prediction error
                err_rc, err_rb, err_lc, err_lb, pres_gt, pres, c_gt, c_pred = centroid_error(outputs, targets, args)
                centroid_pred_err_rt.append(err_rc)
                centroid_pred_err_rb.append(err_rb)
                centroid_pred_err_lt.append(err_lc)
                centroid_pred_err_lb.append(err_lb)
                centroid_pres_err_rt.append(pres_gt[0] ^ pres[0])
                centroid_pres_err_rb.append(pres_gt[1] ^ pres[1])
                centroid_pres_err_lt.append(pres_gt[2] ^ pres[2])
                centroid_pres_err_lb.append(pres_gt[3] ^ pres[3])
            elif 'HRNet' in args.model_type: 
                outputs = F.log_softmax(model(inputs), dim=1)
                # get centroid prediction error
                err_rc, err_rb, err_lc, err_lb, pres_gt, pres, c_gt, c_pred = centroid_error(outputs, targets, args)
                centroid_pred_err_rt.append(err_rc)
                centroid_pred_err_rb.append(err_rb)
                centroid_pred_err_lt.append(err_lc)
                centroid_pred_err_lb.append(err_lb)
                centroid_pres_err_rt.append(pres_gt[0] ^ pres[0])
                centroid_pres_err_rb.append(pres_gt[1] ^ pres[1])
                centroid_pres_err_lt.append(pres_gt[2] ^ pres[2])
                centroid_pres_err_lb.append(pres_gt[3] ^ pres[3])
            elif 'TAPNet' in args.model_type:
                attmaps = inputs[:,3:,:,:]
                inputs = inputs[:,:3,:,:]
                outputs = model(inputs, attmaps)
                # get centroid prediction error
                err_rc, err_rb, err_lc, err_lb, pres_gt, pres = centroid_error(outputs, targets, args)
                centroid_pred_err_rt.append(err_rc)
                centroid_pred_err_rb.append(err_rb)
                centroid_pred_err_lt.append(err_lc)
                centroid_pred_err_lb.append(err_lb)
                centroid_pres_err_rt.append(pres_gt[0] ^ pres[0]) 
                centroid_pres_err_rb.append(pres_gt[1] ^ pres[1])
                centroid_pres_err_lt.append(pres_gt[2] ^ pres[2])
                centroid_pres_err_lb.append(pres_gt[3] ^ pres[3])
            elif 'DeepLab_v3' in args.model_type or 'FCN' in args.model_type:
                outputs = F.log_softmax(model(inputs)['out'], dim=1)
                # get centroid prediction error
                err_rc, err_rb, err_lc, err_lb, pres_gt, pres, c_gt, c_pred = centroid_error(torch.exp(outputs), targets, args)
                centroid_pred_err_rt.append(err_rc)
                centroid_pred_err_rb.append(err_rb)
                centroid_pred_err_lt.append(err_lc)
                centroid_pred_err_lb.append(err_lb)
                centroid_pres_err_rt.append(pres_gt[0] ^ pres[0])
                centroid_pres_err_rb.append(pres_gt[1] ^ pres[1])
                centroid_pres_err_lt.append(pres_gt[2] ^ pres[2])
                centroid_pres_err_lb.append(pres_gt[3] ^ pres[3])
            else:
                raise NotImplementedError
            metrics, metric_dict = get_metrics(outputs, targets, args.metric_fns, args)
            batch_time.update(time.time() - batch_time_start)
            if step % args.save_output_freq == 0:
                disp_image = cv2.imread(str(test_file_names[step]))
                disp_image = cv2.resize(disp_image, (args.input_width, args.input_height))
                output_classes = outputs.data.cpu().numpy().argmax(axis=1)
                mask_array = output_classes[0]
                disp_image = mask_overlay(disp_image, (mask_array==1).astype(np.uint8), color=(255,1,0))
                disp_image = mask_overlay(disp_image, (mask_array==2).astype(np.uint8), color=(255,255,1))
                disp_image = mask_overlay(disp_image, (mask_array==3).astype(np.uint8), color=(0,1,255))
                disp_image = mask_overlay(disp_image, (mask_array==4).astype(np.uint8), color=(0,255,255))
                disp_image = draw_plus(disp_image, [c_gt[0][0],c_gt[1][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[0][1],c_gt[1][1]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[2][0],c_gt[3][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[4][0],c_gt[5][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[4][1],c_gt[5][1]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[6][0],c_gt[7][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_pred[0][0],c_pred[1][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[0][1],c_pred[1][1]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[2][0],c_pred[3][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[4][0],c_pred[5][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[4][1],c_pred[5][1]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[6][0],c_pred[7][0]], color=(255,255,255))
                cv2.imwrite(str(args.output_dir / f'{step}.png'), disp_image)
            idx = 0
            for i, metric_fn in enumerate(args.metric_fns):
                for cls in range(1,args.num_classes):
                    progress_meter_list[idx].update(metrics[i][cls-1], inputs.size(0))
                    idx += 1
                # progress_meter_list[N+3+i].update(metric_dict['metric_'+metric_fn], inputs.size(0))
            if step % args.print_freq == 0:
                progress.display(step, logger=logger)
            step += 1
            data_time_start = time.time()
    
    # compute average detection accuracy
    logger.info(f'Avg. Centroid Detection Error Right Tip: {(1.0-np.mean(centroid_pres_err_rt))*100}')
    logger.info(f'Avg. Centroid Detection Error Right Base: {(1.0-np.mean(centroid_pres_err_rb))*100}')
    logger.info(f'Avg. Centroid Detection Error Left Tip: {(1.0-np.mean(centroid_pres_err_lt))*100}')
    logger.info(f'Avg. Centroid Detection Error Left Base: {(1.0-np.mean(centroid_pres_err_lb))*100}')
    logger.info(f'Std. Centroid Detection Error Right Tip: {np.std(centroid_pres_err_rt)*100}')
    logger.info(f'Std. Centroid Detection Error Right Base: {np.std(centroid_pres_err_rb)*100}')
    logger.info(f'Std. Centroid Detection Error Left Tip: {np.std(centroid_pres_err_lt)*100}')
    logger.info(f'Std. Centroid Detection Error Left Base: {np.std(centroid_pres_err_lb)*100}')

    # compute average centroid error; ignoring nans
    centroid_pred_err_rt = [x for x in centroid_pred_err_rt if not math.isnan(x)]
    centroid_pred_err_rb = [x for x in centroid_pred_err_rb if not math.isnan(x)]
    centroid_pred_err_lt = [x for x in centroid_pred_err_lt if not math.isnan(x)]
    centroid_pred_err_lb = [x for x in centroid_pred_err_lb if not math.isnan(x)]

    logger.info(f'Avg. Centroid Prediction Error Right Tip: {np.mean(centroid_pred_err_rt)} +/- {np.std(centroid_pred_err_rt)}')
    logger.info(f'Avg. Centroid Prediction Error Right Base: {np.mean(centroid_pred_err_rb)} +/- {np.std(centroid_pred_err_rb)}')
    logger.info(f'Avg. Centroid Prediction Error Left Tip: {np.mean(centroid_pred_err_lt)} +/- {np.std(centroid_pred_err_lt)}')
    logger.info(f'Avg. Centroid Prediction Error Left Base: {np.mean(centroid_pred_err_lb)} +/- {np.std(centroid_pred_err_lb)}')
    
    # compute average metrics
    idx = 0
    for i, metric_fn in enumerate(args.metric_fns):
        for cls in range(1,args.num_classes):
            logger.info(f"Avg. {metric_fn} for class {cls}: {progress_meter_list[idx].avg}")
            idx += 1
    # logger.info(f"Metrics: {metrics}")
    # logger.info(f"Avg. Metrics: {metric_dict}")
    return 

def main_worker(args): 
    args.mode = 'testing'
    args.data_dir = Path(args.data_dir)
    args.log_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'logs'))
    args.output_dir = Path(os.path.join(args.expt_savedir, args.expt_name, 'outputs'))
    for dir in [args.log_dir, args.output_dir]:
        if not dir.is_dir():
            print(f"Creating {dir.resolve()} if non-existent")
            dir.mkdir(parents=True, exist_ok=True) 
    
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

    # Set seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")

    # get test dataloader
    if args.dataset=='MICCAI2017':
        test_file_names, _ = get_MICCAI2017_dataset_filenames(args)
    elif args.dataset=='JIGSAWS':
        test_file_names, _ = get_JIGSAWS_dataset_filenames(args)
    elif args.dataset=='MICCAI2015':
        test_file_names, _ = get_MICCAI2015_dataset_filenames(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    # print(test_file_names)
    _, test_dataloader = get_data_loader(args)

    # set up model 
    model = get_model(args)
    if torch.cuda.is_available():
        # model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    else: 
        raise SystemError('GPU device not found! Not configured to train/test.')
    
    # load pre-trained weights if needed
    model, _, load_flag = load_model_weights(model, args.load_wts_model, args.model_type)
    if load_flag:
        logger.info("Model weights loaded from {}".format(args.load_wts_model))
    else: 
        logger.info("No model weights loaded")
    
    if 'TAPNet' in args.model_type: 
        logger.info('Saving attention maps')
        save_attention_maps(model, args)
    test(test_dataloader, model, args, test_file_names, logger, writer=None)
    return

if __name__ == '__main__':
    main()
