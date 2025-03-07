"""
Training script for toolpose segmentation models
Author: Bhargav Ghanekar
"""

import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
sys.path.append('.')
sys.path.append('./models/')
import logging
import json
from pathlib import Path

import configargparse
from configs.config_toolposeseg import train_config_parser as config_parser

import time
import math
import cv2 
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

import tqdm
import matplotlib.pyplot as plt
from src.dataloader import get_data_loader
from src.loss import get_loss
from src.metrics import get_metrics
from models import get_tooltip_segmentation_model as get_model
from utils.dataloader_utils import load_image, load_attmap, get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames
from utils.log_utils import AverageMeter, ProgressMeter, init_logging
from utils.model_utils import load_model_weights, save_model
from utils.train_utils import add_loss_meters, add_metrics_meters

def main():
    parser = configargparse.ArgumentParser()
    parser = config_parser(parser)
    args = parser.parse_args()
    main_worker(args)

def save_attention_maps(model, args):
    model.eval()
    if args.dataset=='MICCAI2017':
        file_names, val_file_names = get_MICCAI2017_dataset_filenames(args)
    elif args.dataset=='JIGSAWS':
        file_names, val_file_names = get_JIGSAWS_dataset_filenames(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized")
    file_names.extend(val_file_names)
    with torch.no_grad():
        for idx, file_name in enumerate(file_names):
            input = torch.from_numpy(load_image(file_name).transpose(2,0,1))
            input = input.type(torch.float32)/255.
            input = transforms.Resize((args.input_height, args.input_width))(input)
            input = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(input)
            input = input.unsqueeze(0)
            input = input.cuda() if torch.cuda.is_available() else input
            if idx%args.num_frames_per_video==0:
                idx_prev = idx
            else: 
                idx_prev = idx-1
            attmap = load_attmap(file_names, idx, args.num_frames_per_video)
            attmap = torch.from_numpy(attmap).unsqueeze(0).unsqueeze(0)
            attmap = transforms.Resize((args.input_height, args.input_width))(attmap)
            attmap = attmap.cuda() if torch.cuda.is_available() else attmap
            output = model(input, attmap)
            output = torch.exp(output) 
            output = torch.sum(output[:,1:,:,:], dim=1, keepdim=False)
            for i in range(input.size(0)):
                cv2.imwrite(str(file_name).replace('images', 'attmaps').replace('jpg', 'png'), (255*output[i].detach().cpu().squeeze().numpy()).astype(np.uint8))
    return

def validate(val_dataloader, model, args, logger, writer, epoch=None):  
    logger.info(f"Validation")
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f')
    progress_meter_list = [batch_time, data_time]
    total_loss = AverageMeter('Total Loss', ':.3f')
    progress_meter_list.append(total_loss)
    progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
    progress_meter_list = add_metrics_meters(progress_meter_list, args.metric_fns, args.num_classes)
    progress = ProgressMeter(len(val_dataloader), progress_meter_list, prefix="Epoch: [{}]".format(epoch))
    model.eval()
    data_time_start = time.time()
    step = 0
    N = len(args.loss_fns)
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            data_time.update(time.time() - data_time_start)
            batch_time_start = time.time()
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            targets = targets.type(torch.LongTensor).cuda() if torch.cuda.is_available() else targets.type(torch.LongTensor) 
            if 'TernausNet' in args.model_type or 'SegFormer' in args.model_type:
                outputs = model(inputs)
            elif 'HRNet' in args.model_type:
                outputs = F.log_softmax(model(inputs), dim=1)
            elif 'TAPNet' in args.model_type:
                targets = targets.squeeze(1)
                outputs = model(inputs[:,:3,:,:], inputs[:,3:,:,:])
            elif 'DeepLab_v3' in args.model_type or 'FCN' in args.model_type:
                outputs = F.log_softmax(model(inputs)['out'], dim=1)
            else:
                raise ValueError(f"Model type {args.model_type} not recognized")
            loss, loss_dict = get_loss(outputs, targets, args.loss_fns, args.loss_wts, args)
            metrics, metric_dict = get_metrics(outputs, targets, args.metric_fns, args)
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                logger.debug(f"Loss is NaN/Inf."); import pdb; pdb.set_trace()
            batch_time.update(time.time() - batch_time_start)
            total_loss.update(loss.item(), inputs.size(0))
            for i, loss_fn in enumerate(args.loss_fns):
                progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], inputs.size(0))
            idx = 0
            for i, metric_fn in enumerate(args.metric_fns):
                for cls in range(1,args.num_classes):
                    progress_meter_list[N+3+idx].update(metrics[i][cls-1], inputs.size(0))
                    idx += 1
                # progress_meter_list[N+3+i].update(metric_dict['metric_'+metric_fn], inputs.size(0))
            if step % args.print_freq == 0:
                progress.display(step, logger=logger)
            step += 1
            data_time_start = time.time()

    writer.add_scalar('Validation/Loss', total_loss.avg, epoch)
    logger.info(f"Validation loss: {total_loss.avg}")
    for i, loss_fn in enumerate(args.loss_fns):
        writer.add_scalar(f'Validation/Loss_{loss_fn}', progress_meter_list[i+3].avg, epoch)
        logger.info(f"Validation loss {loss_fn}: {progress_meter_list[i+3].avg}")
    idx = 0
    for i, metric_fn in enumerate(args.metric_fns):
        for cls in range(1, args.num_classes):
            writer.add_scalar(f'Validation/{metric_fn} {cls}', progress_meter_list[N+3+idx].avg, epoch)
            logger.info(f"Validation metric {metric_fn} {cls}: {progress_meter_list[N+3+idx].avg}")
            idx += 1
    return total_loss.avg

def train_one_epoch(train_dataloader, epoch, model, optimizer, args, logger, writer):
    logger.info(f"Training epoch {epoch}")
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f')
    progress_meter_list = [batch_time, data_time]
    total_loss = AverageMeter('Total Loss', ':.3f')
    progress_meter_list.append(total_loss)
    progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
    progress = ProgressMeter(len(train_dataloader), progress_meter_list, prefix="Epoch: [{}]".format(epoch))
    model.train()
    data_time_start = time.time()
    step = 0
    for inputs, targets in train_dataloader:
        data_time.update(time.time() - data_time_start)
        batch_time_start = time.time()
        inputs = inputs.cuda() if torch.cuda.is_available() else inputs
        targets = targets.type(torch.LongTensor).cuda() if torch.cuda.is_available() else targets.type(torch.LongTensor) 
        optimizer.zero_grad()
        if 'TernausNet' in args.model_type or 'SegFormer' in args.model_type:
            outputs = model(inputs)
        elif 'HRNet' in args.model_type:
            outputs = F.log_softmax(model(inputs), dim=1)
        elif 'TAPNet' in args.model_type:
            targets = targets.squeeze(1)
            outputs = model(inputs[:,:3,:,:], inputs[:,3:,:,:])
        elif 'DeepLab_v3' in args.model_type or 'FCN' in args.model_type:
            outputs = F.log_softmax(model(inputs)['out'], dim=1)
        else:
            raise ValueError(f"Model type {args.model_type} not recognized")
        loss, loss_dict = get_loss(outputs, targets, args.loss_fns, args.loss_wts, args)
        if math.isnan(loss.item()) or math.isinf(loss.item()):
            logger.debug(f"Loss is NaN/Inf."); import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
        
        # display progress
        batch_time.update(time.time() - batch_time_start)
        total_loss.update(loss.item(), inputs.size(0))
        for i, loss_fn in enumerate(args.loss_fns):
            progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], inputs.size(0))
        if step % args.print_freq == 0:
            progress.display(step, logger=logger)
        step += 1
        data_time_start = time.time()

    writer.add_scalar('Training/Loss', total_loss.avg, epoch)
    logger.info(f"Training loss: {total_loss.avg}")
    for i, loss_fn in enumerate(args.loss_fns):
        writer.add_scalar(f'Training/Loss_{loss_fn}', progress_meter_list[i+3].avg, epoch)
        logger.info(f"Training loss {loss_fn}: {progress_meter_list[i+3].avg}")
    return model, total_loss.avg

def main_worker(args):
    assert len(args.loss_fns) == len(args.loss_wts), "Number of loss functions and loss weights should be equal"
    assert len(args.loss_fns) > 0, "At least one loss function should be specified"
    assert len(args.class_weights) == args.num_classes, "Number of class weights should be equal to number of classes"
    args.class_weights = np.array(args.class_weights)
    args.expt_name = '_'.join([args.expt_name, str(args.fold_index) if args.fold_index!=-1 else 'full']) 
    args.data_dir = Path(args.data_dir)
    args.log_dir = Path(os.path.join(args.expt_savedir, args.expt_name, "logs"))
    args.output_dir = Path(os.path.join(args.expt_savedir, args.expt_name, "outputs"))
    args.ckpt_dir = Path(os.path.join(args.expt_savedir, args.expt_name, "ckpts"))
    for dir in [args.log_dir, args.output_dir, args.ckpt_dir]:
        if not dir.is_dir():
            print(f"Creating {dir.resolve()} if non-existent")
            dir.mkdir(parents=True, exist_ok=True)
    writer, logger = init_logging(args)
    logger.info("Code files copied to log directory")

    # Set seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")

    # set up dataloaders
    train_dataloader, val_dataloader = get_data_loader(args)

    # set up model 
    model = get_model(args)
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    else: 
        raise SystemError('GPU device not found! Not configured to train/test.')
    
    # load pre-trained weights if needed
    model, start_epoch, load_flag = load_model_weights(model, args.load_wts_model, args.model_type)
    if load_flag:
        logging.info("Model weights loaded from {}".format(args.load_wts_model))
    else: 
        logging.info("No model weights loaded")
    
    # set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler=='StepDecay':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.num_epochs/2), gamma=0.1)
    elif args.scheduler=='Constant':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epochs, gamma=1.0)
    else: 
        raise ValueError('Unknown scheduler: {}'.format(args.scheduler))
    logging.info("Learning rate {:.6f}".format(args.lr))

    # set up epoch iterations
    if args.resume: 
        start_epoch = args.starting_epoch
    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        random.seed()
        try: 
            logging.info('Training, current LR: {}'.format(scheduler.get_last_lr()))
            model, loss = train_one_epoch(train_dataloader, epoch, model, optimizer, args, logger, writer)
            logging.info("Epoch: {}, Loss: {:.5f}".format(epoch, loss))
            scheduler.step()
            metrics = validate(val_dataloader, model, args, logger, writer, epoch)
            logging.info(json.dumps(metrics))
            if 'TAPNet' in args.model_type and args.update_attmaps:
                logging.info('Saving attention maps')
                save_attention_maps(model, args)
            if epoch%args.save_freq==0:
                save_model(model, args.ckpt_dir, optimizer=optimizer, epoch=epoch)
        except KeyboardInterrupt: 
            logging.info('Ctrl+C, saving snapshot')
            save_model(model, args.ckpt_dir, optimizer=optimizer, epoch=epoch)
            logging.info('Done.')
            return 


if __name__ == '__main__':
    main()
