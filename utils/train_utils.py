from utils.log_utils import AverageMeter

def add_loss_meters(progress_meter_list, loss_fns):
    for loss_fn in loss_fns:
        progress_meter_list.append(AverageMeter(f'{loss_fn} Loss', ':.3f'))
    return progress_meter_list

def add_metrics_meters(progress_meter_list, metric_fns, num_classes):
    for metric_fn in metric_fns: 
        for cls in range(num_classes):
            if cls==0:
                continue
            progress_meter_list.append(AverageMeter(f'{metric_fn} Metric Class {cls}', ':.3f'))
    return progress_meter_list
