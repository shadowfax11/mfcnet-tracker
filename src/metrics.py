import numpy as np 
import torch 

def get_metrics(outputs, targets, metric_fns, args):
    metric_dict = {}
    output_classes = outputs.data.cpu().numpy().argmax(axis=1)
    target_classes = targets.data.cpu().numpy()
    confusion_matrix = calculate_confusion_matrix_from_arrays(output_classes, target_classes, args.num_classes)
    confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
    metric_vals_per_class = []
    for metric_fn in metric_fns:
        if metric_fn == 'jaccard':
            raise NotImplementedError
        elif metric_fn == 'iou':
            ious = {}
            iou_list = []
            for cls in range(args.num_classes):
                if cls==0:
                    continue # exclude background
                iou = get_jaccard(target_classes==cls, output_classes==cls)
                ious['iou_{}'.format(cls)] = iou
                iou_list.append(iou)
            metric = np.mean(list(ious.values()))
            metric_vals_per_class.append(iou_list)
        elif metric_fn == 'dice':
            dices = {}
            dice_list = []
            for cls in range(args.num_classes):
                if cls==0:
                    continue # exclude background
                dice = get_dice(target_classes==cls, output_classes==cls)
                dices['dice_{}'.format(cls)] = dice
                dice_list.append(dice)
            metric = np.mean(list(dices.values()))
            metric_vals_per_class.append(dice_list)
        else:
            raise ValueError(f'Metric function {metric_fn} not implemented')
        metric_dict['metric_' + metric_fn] = metric
    return metric_vals_per_class, metric_dict

def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(axis=-2).sum(axis=-1)
    union = y_true.sum(axis=-2).sum(axis=-1) + y_pred.sum(axis=-2).sum(axis=-1)
    return ((intersection + epsilon) / (union - intersection + epsilon))[0]

def get_dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten())).T
    confusion_matrix, _ = np.histogramdd(replace_indices,
                                        bins=(nr_labels, nr_labels),
                                        range=[(0, nr_labels), (0, nr_labels)])
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

