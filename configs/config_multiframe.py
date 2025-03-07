"""
Configuration file stating all the args for multi-frame segmentation task
"""

def train_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--data_dir', type=str, default='/home/bg40/surgical_video_datasets/miccai2017/', 
                        help='Path to data directory. Default: /home/bg40/surgical_video_datasets/miccai2017/')
    parser.add_argument('--dataset', type=str, default='MICCAI2017', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS'],
                        help='Dataset name. Default: MICCAI2017')
    parser.add_argument('--fold_index', type=int, default=-1, choices=[-1,0,1,2,3], 
                        help='Fold index for cross validation. Default: -1, no cross validation')
    parser.add_argument('--prediction_task', type=str, default='toolpose_segmentation', 
                        choices=['tooltip_segmentation', 'toolpose_segmentation', 'endovis15_segmentation', 'binary'],
                        help='Prediction task. Default: toolpose_segmentation')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'], 
                        help='Mode of operation. Default: training')
    parser.add_argument('--num_frames_per_video', type=int, default=225, 
                        help='Number of frames per video/folder in the dataset. Default: 225')
    parser.add_argument('--num_input_frames', type=int, default=3,
                        help='Number of input frames for the model. Default: 3')
    
    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='./', 
                        help='Path to save experiment results. Default: ./')
    parser.add_argument('--expt_name', type=str, default='multiframe_segmentation_expt',
                        help='Experiment name. Default: multiframe_segmentation_expt')
    parser.add_argument('--print_freq', type=int, default=10, 
                        help='Print frequency. Default: 10')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency. Default: 10')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

    # optimizer related arguments   
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size. Default: 8')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader. Default: 12')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes (incl. background). Default: 5')
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice'], choices=['iou', 'dice'], 
                        help='List of metric functions. Default: iou, dice')
    parser.add_argument('--loss_fns', type=str, nargs='+', default=['nll'], choices=['mse', 'nll', 'soft_jaccard'],  
                        help='List of loss functions. Default: nll')
    parser.add_argument('--loss_wts', type=float, nargs='+', default=[1.0], 
                        help='List of loss weights. Default: 1.0')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate. Default: 1e-4')
    parser.add_argument('--scheduler', type=str, default='StepDecay', choices=['StepDecay', 'Constant'], 
                        help='Learning rate scheduler. Default: StepDecay at halfway point')
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='Number of epochs. Default: 10')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed for random number generator. Default: 42')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='Resume training')
    parser.add_argument('--starting_epoch', type=int, default=0, 
                        help='Starting epoch. Default: 0')
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1,100,100,100,100],
                        help='Class weights for NLL loss function. Default: [1,100,100,100,100]')
    
    # model related arguments
    parser.add_argument('--model_type', type=str, default='FCNMulti-Basic', 
                        choices=['TernausNetMulti-Basic', 'TernausNetMulti-Large', 'DeepLabMulti-Basic', 'DeepLabMulti-Large', 
                                 'FCNMulti-Basic', 'FCNMulti-Large', 'SegFormerMulti-Basic', 'SegFormerMulti-Large', 'HRNetMulti-Basic', 'HRNetMulti-Large'], 
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='Use pre-trained weights. Default: False')
    parser.add_argument('--train_base_model', type=bool, default=False,
                        help='Train base model. Default: False')
    parser.add_argument('--load_wts_base_model', type=str, default=None,
                        help='Path to base model weights from a pretrained per-frame model. Default: None')
    parser.add_argument('--load_wts_model', type=str, default=None, 
                        help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=1024, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=1280, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_model', type=str, default='RAFT', choices=['RAFT', 'FlowFormerPlusPlus'],)
    parser.add_argument('--add_depth_inputs', type=bool, default=False, help='Add monocular depth inputs')
    return parser

def test_config_parser(parser):
    # dataset related arguments
    parser.add_argument('--data_dir', type=str, default='/home/bg40/surgical_video_datasets/miccai2017/', 
                        help='Path to data directory. Default: /home/bg40/surgical_video_datasets/miccai2017/')
    parser.add_argument('--dataset', type=str, default='MICCAI2017', choices=['MICCAI2015', 'MICCAI2017', 'JIGSAWS'],
                        help='Dataset name. Default: MICCAI2017')
    parser.add_argument('--prediction_task', type=str, default='toolpose_segmentation', 
                        choices=['tooltip_segmentation', 'toolpose_segmentation', 'endovis15_segmentation', 'binary'], 
                        help='Prediction task. Default: toolpose_segmentation')
    parser.add_argument('--num_frames_per_video', type=int, default=75, 
                        help='Number of frames per video/folder in the dataset. Default: 75')
    parser.add_argument('--num_input_frames', type=int, default=3,
                        help='Number of input frames for the model. Default: 3')
    
    # I/O related arguments
    parser.add_argument('--expt_savedir', type=str, default='./', 
                        help='Path to save experiment results. Default: ./')
    parser.add_argument('--expt_name', type=str, default='multiframe_segmentation_expt',
                        help='Experiment name. Default: multiframe_segmentation_expt')
    parser.add_argument('--print_freq', type=int, default=10, 
                        help='Print frequency. Default: 10')
    parser.add_argument('--save_output_freq', type=int, default=10, 
                        help='Save output frequency. Default: 10')

    # optimizer related arguments   
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes (incl. background). Default: 5')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloader. Default: 12')
    parser.add_argument('--metric_fns', type=str, nargs='+', default=['iou', 'dice'], choices=['iou', 'dice'], 
                        help='List of metric functions. Default: iou, dice')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed for random number generator. Default: 42')
    parser.add_argument('--resume', type=bool, default=False, 
                        help='Resume training. Default: False')
    
    # model related arguments
    parser.add_argument('--model_type', type=str, default='FCNMulti-Basic', 
                        choices=['TernausNetMulti-Basic', 'TernausNetMulti-Large', 'DeepLabMulti-Basic', 'DeepLabMulti-Large', 
                                 'FCNMulti-Basic', 'FCNMulti-Large', 'SegFormerMulti-Basic', 'SegFormerMulti-Large', 'HRNetMulti-Basic', 'HRNetMulti-Large'],  
                        help='Model name')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='Use pre-trained weights. Default: False')
    parser.add_argument('--load_wts_base_model', type=str, default=None,
                        help='Path to base model weights from a pretrained per-frame model. Default: None')
    parser.add_argument('--load_wts_model', type=str, default=None, 
                        help='Path to model weights. Default: None')
    parser.add_argument('--input_height', type=int, default=1024, help='NN input image height')
    parser.add_argument('--input_width', type=int, default=1280, help='NN input image width')
    parser.add_argument('--add_optflow_inputs', type=bool, default=False, help='Add optical flow inputs')
    parser.add_argument('--optflow_model', type=str, default='RAFT', choices=['RAFT', 'FlowFormerPlusPlus'],)
    parser.add_argument('--add_depth_inputs', type=bool, default=False, help='Add monocular depth inputs')
    return parser

