"""
Training script for multi-frame tool-tip/pose detection. 
Author: Bhargav Ghanekar
"""

import os
# import torch
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['OMP_NUM_THREADS'] = '12'
# os.environ['MKL_NUM_THREADS'] = '12'
# os.environ['OPENBLAS_NUM_THREADS'] = '12'
# torch.set_num_threads(12)
import sys
sys.path.append('.')
sys.path.append('./models/')
import logging, json, configargparse
from pathlib import Path

from configs.config_multiframe import train_config_parser as config_parser

import tqdm, time, math, random
import cv2 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

import matplotlib.pyplot as plt
from src.dataloader_multiframe import get_data_loader
from src.engine import train_one_epoch, validate
from models import get_multiframe_segmentation_model as get_model
from utils.log_utils import AverageMeter, ProgressMeter, init_logging
from utils.model_utils import load_model_weights, save_model
from utils.train_utils import add_loss_meters, add_metrics_meters

def main():
    parser = configargparse.ArgumentParser()
    parser = config_parser(parser)
    args = parser.parse_args()
    main_worker(args)

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

    # set up optical flow model if needed
    if args.add_optflow_inputs:
        if args.optflow_model=='RAFT':
            from torchvision.models.optical_flow import raft_large
            optflow_model = raft_large(pretrained=True, progress=False)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    optflow_model = nn.DataParallel(optflow_model)
                optflow_model = optflow_model.cuda()
        elif args.optflow_model=='FlowFormerPlusPlus':
            sys.path.append('./models/optical_flow/flowformerplusplus')
            sys.path.append('./models/optical_flow/flowformerplusplus/PerCostFormer3')
            from models.optical_flow.flowformerplusplus.ffpp_cfg_things import get_cfg
            from models.optical_flow.flowformerplusplus import build_flowformer
            cfg = get_cfg()
            optflow_model = build_flowformer(cfg)
            state_dict = torch.load('./models/optical_flow/flowformerplusplus/ckpts/ffpp_things.pth')
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            optflow_model.load_state_dict(new_state_dict)
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    optflow_model = nn.DataParallel(optflow_model)
                optflow_model = optflow_model.cuda()
        else: 
            raise SystemError('GPU device not found! Not configured to train/test.')
        optflow_model.eval()
        logger.info(f"{args.optflow_model} optical flow model loaded")
    else: 
        optflow_model = None

    # set up model 
    model = get_model(args)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
    else: 
        raise SystemError('GPU device not found! Not configured to train/test.')
    
    if args.load_wts_base_model is not None:
        basemodel_state = torch.load(args.load_wts_base_model)
        model.base_model.load_state_dict(basemodel_state['model'])
        logger.info("Base model weights loaded from {}".format(args.load_wts_base_model))

    # load pre-trained weights if needed
    model, start_epoch, load_flag = load_model_weights(model, args.load_wts_model, args.model_type)
    if load_flag:
        logging.info("Model weights loaded from {}".format(args.load_wts_model))
    else: 
        logging.info("No model weights loaded")
    
    # set up optimizer and scheduler
    if args.train_base_model:
        logger.info("Training base model and multi-frame network")
        if torch.cuda.device_count() > 1:
            if args.load_wts_base_model is not None:
                optimizer = optim.Adam([{'params': model.module.base_model.parameters(), 'lr': args.lr/(100*args.num_input_frames)}, 
                                {'params': model.module.multiframe_net.parameters()}], lr=args.lr)
            else: 
                optimizer = optim.Adam([{'params': model.module.base_model.parameters(), 'lr': args.lr/args.num_input_frames}, 
                                {'params': model.module.multiframe_net.parameters()}], lr=args.lr)
        else: 
            if args.load_wts_base_model is not None:
                optimizer = optim.Adam([{'params': model.base_model.parameters(), 'lr': args.lr/(100*args.num_input_frames)}, 
                                {'params': model.multiframe_net.parameters()}], lr=args.lr)
            else: 
                optimizer = optim.Adam([{'params': model.base_model.parameters(), 'lr': args.lr/args.num_input_frames}, 
                                {'params': model.multiframe_net.parameters()}], lr=args.lr)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        logger.info("Training multi-frame network only")
        model.base_model.eval()
        for param in model.base_model.parameters():
            param.requires_grad = False
        model.multiframe_net.train()
        optimizer = optim.Adam(model.multiframe_net.parameters(), lr=args.lr)
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
        if args.train_base_model:
            model.train()
        else:
            model.base_model.eval()
            model.multiframe_net.train()
        random.seed()
        try: 
            logging.info('Training, current LR: {}'.format(scheduler.get_last_lr()))
            model, loss = train_one_epoch(train_dataloader, epoch, model, optimizer, args, logger, writer, optflow_model=optflow_model)
            logging.info("Epoch: {}, Loss: {:.5f}".format(epoch, loss))
            scheduler.step()
            metrics = validate(val_dataloader, model, args, logger, writer, epoch, optflow_model=optflow_model)
            logging.info(json.dumps(metrics))
            if epoch%args.save_freq==0:
                save_model(model, args.ckpt_dir, optimizer=optimizer, epoch=epoch)
        except KeyboardInterrupt: 
            logging.info('Ctrl+C, saving snapshot')
            save_model(model, args.ckpt_dir, optimizer=optimizer, epoch=epoch)
            logging.info('Done.')
            return 

if __name__ == '__main__':
    main()
