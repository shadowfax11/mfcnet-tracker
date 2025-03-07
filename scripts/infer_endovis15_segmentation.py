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
from utils.localization_utils_v2 import centroid_error_10_classes
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def main(): 
    parser = configargparse.ArgumentParser() 
    parser = config_parser(parser) 
    args = parser.parse_args() 
    main_worker(args) 

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
    centroid_pred_err_r1 = []; centroid_pred_err_r2 = []; centroid_pred_err_r3 = []; centroid_pred_err_r4 = []; centroid_pred_err_r5 = []
    centroid_pred_err_l1 = []; centroid_pred_err_l2 = []; centroid_pred_err_l3 = []; centroid_pred_err_l4 = []; centroid_pred_err_l5 = []
    centroid_pres_err_r1 = []; centroid_pres_err_r2 = []; centroid_pres_err_r3 = []; centroid_pres_err_r4 = []; centroid_pres_err_r5 = []
    centroid_pres_err_l1 = []; centroid_pres_err_l2 = []; centroid_pres_err_l3 = []; centroid_pres_err_l4 = []; centroid_pres_err_l5 = []
    all_pres_gt = []; all_pres = []

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
                err, pres_gt, pres, c_gt, c_pred = centroid_error_10_classes(outputs, targets)
                centroid_pred_err_r1.append(err[0]); centroid_pred_err_r2.append(err[1]); centroid_pred_err_r3.append(err[2]); centroid_pred_err_r4.append(err[3]); centroid_pred_err_r5.append(err[4])
                centroid_pred_err_l1.append(err[5]); centroid_pred_err_l2.append(err[6]); centroid_pred_err_l3.append(err[7]); centroid_pred_err_l4.append(err[8]); centroid_pred_err_l5.append(err[9])
                centroid_pres_err_r1.append(pres_gt[0]^pres[0]); centroid_pres_err_r2.append(pres_gt[1]^pres[1]); centroid_pres_err_r3.append(pres_gt[2]^pres[2]); centroid_pres_err_r4.append(pres_gt[3]^pres[3]); centroid_pres_err_r5.append(pres_gt[4]^pres[4])
                centroid_pres_err_l1.append(pres_gt[5]^pres[5]); centroid_pres_err_l2.append(pres_gt[6]^pres[6]); centroid_pres_err_l3.append(pres_gt[7]^pres[7]); centroid_pres_err_l4.append(pres_gt[8]^pres[8]); centroid_pres_err_l5.append(pres_gt[9]^pres[9])
            elif 'HRNet' in args.model_type: 
                outputs = F.log_softmax(model(inputs), dim=1)
                # get centroid prediction error
                err, pres_gt, pres, c_gt, c_pred = centroid_error_10_classes(outputs, targets)
                centroid_pred_err_r1.append(err[0]); centroid_pred_err_r2.append(err[1]); centroid_pred_err_r3.append(err[2]); centroid_pred_err_r4.append(err[3]); centroid_pred_err_r5.append(err[4])
                centroid_pred_err_l1.append(err[5]); centroid_pred_err_l2.append(err[6]); centroid_pred_err_l3.append(err[7]); centroid_pred_err_l4.append(err[8]); centroid_pred_err_l5.append(err[9])
                centroid_pres_err_r1.append(pres_gt[0]^pres[0]); centroid_pres_err_r2.append(pres_gt[1]^pres[1]); centroid_pres_err_r3.append(pres_gt[2]^pres[2]); centroid_pres_err_r4.append(pres_gt[3]^pres[3]); centroid_pres_err_r5.append(pres_gt[4]^pres[4])
                centroid_pres_err_l1.append(pres_gt[5]^pres[5]); centroid_pres_err_l2.append(pres_gt[6]^pres[6]); centroid_pres_err_l3.append(pres_gt[7]^pres[7]); centroid_pres_err_l4.append(pres_gt[8]^pres[8]); centroid_pres_err_l5.append(pres_gt[9]^pres[9])
            elif 'TAPNet' in args.model_type:
                raise NotImplementedError
            elif 'DeepLab_v3' in args.model_type or 'FCN' in args.model_type:
                outputs = F.log_softmax(model(inputs)['out'], dim=1)
                # get centroid prediction error
                err, pres_gt, pres, c_gt, c_pred = centroid_error_10_classes(outputs, targets)
                centroid_pred_err_r1.append(err[0]); centroid_pred_err_r2.append(err[1]); centroid_pred_err_r3.append(err[2]); centroid_pred_err_r4.append(err[3]); centroid_pred_err_r5.append(err[4])
                centroid_pred_err_l1.append(err[5]); centroid_pred_err_l2.append(err[6]); centroid_pred_err_l3.append(err[7]); centroid_pred_err_l4.append(err[8]); centroid_pred_err_l5.append(err[9])
                centroid_pres_err_r1.append(pres_gt[0]^pres[0]); centroid_pres_err_r2.append(pres_gt[1]^pres[1]); centroid_pres_err_r3.append(pres_gt[2]^pres[2]); centroid_pres_err_r4.append(pres_gt[3]^pres[3]); centroid_pres_err_r5.append(pres_gt[4]^pres[4])
                centroid_pres_err_l1.append(pres_gt[5]^pres[5]); centroid_pres_err_l2.append(pres_gt[6]^pres[6]); centroid_pres_err_l3.append(pres_gt[7]^pres[7]); centroid_pres_err_l4.append(pres_gt[8]^pres[8]); centroid_pres_err_l5.append(pres_gt[9]^pres[9])
            else:
                raise NotImplementedError
            all_pres_gt.append(pres_gt); all_pres.append(pres)
            metrics, metric_dict = get_metrics(outputs, targets, args.metric_fns, args)
            batch_time.update(time.time() - batch_time_start)
            if step % args.save_output_freq == 0:
                disp_image = cv2.imread(str(test_file_names[step]))
                disp_image = cv2.resize(disp_image, (args.input_width, args.input_height))
                output_classes = outputs.data.cpu().numpy().argmax(axis=1)
                mask_array = output_classes[0]
                disp_image = mask_overlay(disp_image, (mask_array==1).astype(np.uint8), color=(0,63,0), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==2).astype(np.uint8), color=(0,127,0), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==3).astype(np.uint8), color=(0,255,0), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==4).astype(np.uint8), color=(0,1,255), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==5).astype(np.uint8), color=(255,1,0), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==6).astype(np.uint8), color=(0,63,0), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==7).astype(np.uint8), color=(0,127,0), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==8).astype(np.uint8), color=(0,255,0), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==9).astype(np.uint8), color=(0,1,255), wt=0.9)
                disp_image = mask_overlay(disp_image, (mask_array==10).astype(np.uint8), color=(255,1,0), wt=0.9)
                
                disp_image = draw_plus(disp_image, [c_gt[0][0],c_gt[1][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[2][0],c_gt[3][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[4][0],c_gt[5][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[6][0],c_gt[7][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[8][0],c_gt[9][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[10][0],c_gt[11][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[12][0],c_gt[13][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[14][0],c_gt[15][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[16][0],c_gt[17][0]], color=(0,255,0))
                disp_image = draw_plus(disp_image, [c_gt[18][0],c_gt[19][0]], color=(0,255,0))
                
                disp_image = draw_plus(disp_image, [c_pred[0][0],c_pred[1][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[2][0],c_pred[3][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[4][0],c_pred[5][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[6][0],c_pred[7][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[8][0],c_pred[9][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[10][0],c_pred[11][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[12][0],c_pred[13][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[14][0],c_pred[15][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[16][0],c_pred[17][0]], color=(255,255,255))
                disp_image = draw_plus(disp_image, [c_pred[18][0],c_pred[19][0]], color=(255,255,255))
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
    
    # Convert the lists to numpy arrays for easier handling
    all_pres_gt = np.array(all_pres_gt)
    all_pres = np.array(all_pres)

    # Initialize an empty confusion matrix for each class
    conf_matrices = []
    precisions = []
    recalls = []

    for i in range(args.num_classes-1):
        # Compute confusion matrix for class i (binary: present/not present)
        cm = confusion_matrix(all_pres_gt[:, i], all_pres[:, i], labels=[0, 1])
        conf_matrices.append(cm)
        # Calculate precision and recall for class i
        # precision = precision_score(all_pres_gt[:, i], all_pres[:, i])
        # recall = recall_score(all_pres_gt[:, i], all_pres[:, i])
        # precisions.append(precision)
        # recalls.append(recall)

    # Output the results for each class
    for i in range(int((args.num_classes-1)/2)):
        cm = conf_matrices[i]+conf_matrices[i+5]
        true_pos = np.diag(cm)
        false_pos = np.sum(cm, axis=0) - true_pos
        false_neg = np.sum(cm, axis=1) - true_pos
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        accuracy = np.sum(true_pos) / np.sum(cm)
        logger.info(f"Class {i+1}:")
        # logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Precision: {100*precision[1]}")
        logger.info(f"Recall: {100*recall[1]}")
        logger.info(f"Accuracy: {100*accuracy}")
        # logger.info("\n")
    
    # compute average centroid error; ignoring nans
    centroid_pred_err_r1 = [x for x in centroid_pred_err_r1 if not math.isnan(x)]
    centroid_pred_err_r2 = [x for x in centroid_pred_err_r2 if not math.isnan(x)]
    centroid_pred_err_r3 = [x for x in centroid_pred_err_r3 if not math.isnan(x)]
    centroid_pred_err_r4 = [x for x in centroid_pred_err_r4 if not math.isnan(x)]
    centroid_pred_err_r5 = [x for x in centroid_pred_err_r5 if not math.isnan(x)]
    centroid_pred_err_l1 = [x for x in centroid_pred_err_l1 if not math.isnan(x)]
    centroid_pred_err_l2 = [x for x in centroid_pred_err_l2 if not math.isnan(x)]
    centroid_pred_err_l3 = [x for x in centroid_pred_err_l3 if not math.isnan(x)]
    centroid_pred_err_l4 = [x for x in centroid_pred_err_l4 if not math.isnan(x)]
    centroid_pred_err_l5 = [x for x in centroid_pred_err_l5 if not math.isnan(x)]
    logger.info(f'Avg. Centroid Prediction Error Class 1: {np.mean(centroid_pred_err_r1+centroid_pred_err_l1)} +/- {np.std(centroid_pred_err_r1+centroid_pred_err_l1)}')
    logger.info(f'Avg. Centroid Prediction Error Class 2: {np.mean(centroid_pred_err_r2+centroid_pred_err_l2)} +/- {np.std(centroid_pred_err_r2+centroid_pred_err_l2)}')
    logger.info(f'Avg. Centroid Prediction Error Class 3: {np.mean(centroid_pred_err_r3+centroid_pred_err_l3)} +/- {np.std(centroid_pred_err_r3+centroid_pred_err_l3)}')
    logger.info(f'Avg. Centroid Prediction Error Class 4: {np.mean(centroid_pred_err_r4+centroid_pred_err_l4)} +/- {np.std(centroid_pred_err_r4+centroid_pred_err_l4)}')
    logger.info(f'Avg. Centroid Prediction Error Class 5: {np.mean(centroid_pred_err_r5+centroid_pred_err_l5)} +/- {np.std(centroid_pred_err_r5+centroid_pred_err_l5)}')
    m1 = np.mean(centroid_pred_err_r1+centroid_pred_err_l1); s1 = np.std(centroid_pred_err_r1+centroid_pred_err_l1)
    m2 = np.mean(centroid_pred_err_r2+centroid_pred_err_l2); s2 = np.std(centroid_pred_err_r2+centroid_pred_err_l2)
    m3 = np.mean(centroid_pred_err_r3+centroid_pred_err_l3); s3 = np.std(centroid_pred_err_r3+centroid_pred_err_l3)
    m4 = np.mean(centroid_pred_err_r4+centroid_pred_err_l4); s4 = np.std(centroid_pred_err_r4+centroid_pred_err_l4)
    m5 = np.mean(centroid_pred_err_r5+centroid_pred_err_l5); s5 = np.std(centroid_pred_err_r5+centroid_pred_err_l5)
    logger.info(f'Avg. Centroid Prediction Error: {np.mean([m1,m2,m3,m4,m5])} +/- {np.mean([s1,s2,s3,s4,s5])}')
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
        raise NotImplementedError
    test(test_dataloader, model, args, test_file_names, logger, writer=None)
    return

if __name__ == '__main__':
    main()
