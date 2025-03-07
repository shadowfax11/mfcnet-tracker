import time, math, random, os, sys
import logging
import torch 
import torch.nn.functional as F
from src.loss import get_loss 
from src.metrics import get_metrics 
sys.path.append('../utils/')
from utils.log_utils import AverageMeter, ProgressMeter 
from utils.train_utils import add_loss_meters, add_metrics_meters 

def train_one_epoch(dataloader, epoch, model, optimizer, args, logger, writer=None, optflow_model=None):
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    batch_time = AverageMeter('Time', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    progress_meter_list = [batch_time, data_time] 
    total_loss = AverageMeter('Total Loss', ':.3f')
    progress_meter_list.append(total_loss) 
    progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix=f"Epoch: [{epoch}]")
    if args.train_base_model:
        model.train()
    else:
        model.base_model.eval()
        model.multiframe_net.train()
    data_time_start = time.time()
    step = 0 
    for sample in dataloader: 
        data_time.update(time.time() - data_time_start)
        batch_time_start = time.time() 
        if torch.cuda.is_available():
            input = [sample['input'][i].cuda(non_blocking=True) for i in range(len(sample['input']))]
            mask = sample['mask'].type(torch.LongTensor).cuda(non_blocking=True)
            if args.add_depth_inputs: 
                input_depth = [sample['input_depth'][i].cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
        else: 
            mask = sample['mask'].type(torch.LongTensor)
        if args.add_optflow_inputs:
            optflow = []
            frame0 = F.interpolate(input[0], scale_factor=1.0, mode='nearest')
            if args.optflow_model == 'FlowFormerPlusPlus':
                frame0 = frame0 * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
            for i in range(1,len(input)): 
                frame = F.interpolate(input[i], scale_factor=1.0, mode='nearest')
                if args.optflow_model == 'FlowFormerPlusPlus':
                    frame = frame * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
                if 'Basic' in args.model_type:
                    flow = optflow_model(frame, frame0)[-1]
                else: 
                    flow = optflow_model(frame0, frame)[-1]
                flow = F.interpolate(flow/1.0, size=(input[0].size(2), input[0].size(3)), mode='bilinear', align_corners=True)
                optflow.append(flow)
        optimizer.zero_grad()
        mask = mask.squeeze(1)
        if args.add_optflow_inputs:
            if args.add_depth_inputs:
                output = model(input, optflow=optflow, depth=input_depth)
            else:
                output = model(input, optflow=optflow)
        elif args.add_depth_inputs:
            output = model(input, depth=input_depth)
        else: 
            output = model(input)
        output = F.log_softmax(output, dim=1)
        loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args)
        if math.isnan(loss.item()) or math.isinf(loss.item()): 
            logger.debug(f"Loss is {loss.item()}. Exiting...")
            import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
        # display progress
        batch_time.update(time.time() - batch_time_start)
        total_loss.update(loss.item(), input[0].size(0))
        for i, loss_fn in enumerate(args.loss_fns):
            progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], input[0].size(0))
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

def validate(dataloader, model, args, logger, writer=None, epoch=None, optflow_model=None):
    if args.add_optflow_inputs:
        assert optflow_model is not None, "Optical flow model should be provided"
        optflow_model.eval()
    batch_time = AverageMeter(' Forward Time', ':2.2f')
    data_time = AverageMeter(' Data Time', ':2.2f')
    progress_meter_list = [batch_time, data_time]
    total_loss = AverageMeter('Total Loss', ':.3f')
    progress_meter_list.append(total_loss)
    progress_meter_list = add_loss_meters(progress_meter_list, args.loss_fns)
    progress_meter_list = add_metrics_meters(progress_meter_list, args.metric_fns, args.num_classes)
    progress = ProgressMeter(len(dataloader), progress_meter_list, prefix='Epoch: [{}]'.format(epoch))
    model.eval()
    data_time_start = time.time() 
    step = 0
    N = len(args.loss_fns) 
    with torch.no_grad(): 
        for sample in dataloader: 
            data_time.update(time.time() - data_time_start)
            batch_time_start = time.time() 
            if torch.cuda.is_available():
                input = [sample['input'][i].cuda(non_blocking=True) for i in range(len(sample['input']))]
                mask = sample['mask'].type(torch.LongTensor).cuda(non_blocking=True)
                if args.add_depth_inputs:
                    input_depth = [sample['input_depth'][i].cuda(non_blocking=True) for i in range(len(sample['input_depth']))]
            else: 
                mask = sample['mask'].type(torch.LongTensor).cuda(non_blocking=True)
            if args.add_optflow_inputs:
                optflow = []
                frame0 = F.interpolate(input[0], scale_factor=1.0, mode='nearest')
                if args.optflow_model == 'FlowFormerPlusPlus':
                    frame0 = frame0 * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
                for i in range(1,len(input)): 
                    frame = F.interpolate(input[i], scale_factor=1.0, mode='nearest')
                    if args.optflow_model == 'FlowFormerPlusPlus':
                        frame = frame * 0.225 / 0.5 # approximate scaling so as to match the input range of FlowFormerPlusPlus
                    if 'Basic' in args.model_type:
                        flow = optflow_model(frame, frame0)[-1]
                    else: 
                        flow = optflow_model(frame0, frame)[-1]
                    flow = F.interpolate(flow/1.0, size=(input[0].size(2), input[0].size(3)), mode='bilinear', align_corners=True)
                    optflow.append(flow)
            mask = mask.squeeze(1)
            if args.add_optflow_inputs:
                if args.add_depth_inputs:
                    output = model(input, optflow=optflow, depth=input_depth)
                else:
                    output = model(input, optflow=optflow)
            elif args.add_depth_inputs:
                output = model(input, depth=input_depth)
            else:
                output = model(input)
            output = F.log_softmax(output, dim=1)
            loss, loss_dict = get_loss(output, mask, args.loss_fns, args.loss_wts, args)
            metrics, metric_dict = get_metrics(output, mask, args.metric_fns, args)
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                logger.debug(f"Loss is NaN/Inf."); import pdb; pdb.set_trace()
            batch_time.update(time.time() - batch_time_start)
            total_loss.update(loss.item(), input[0].size(0))
            for i, loss_fn in enumerate(args.loss_fns):
                progress_meter_list[i+3].update(loss_dict['loss_'+loss_fn], input[0].size(0))
            idx = 0
            for i, metric_fn in enumerate(args.metric_fns):
                for cls in range(1, args.num_classes):
                    progress_meter_list[N+3+idx].update(metrics[i][cls-1], input[0].size(0))
                    idx += 1
                # progress_meter_list[N+3+i].update(metric_dict['metric_'+metric_fn], input[0].size(0))
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
