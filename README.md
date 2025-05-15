# mfcnet-tracker
Official code repository for Ghanekar et al. 2025 "Video-based Surgical Tool-tip and Keypoint Tracking using Multi-frame Context-driven Deep Learning Models" [Accepted to IEEE ISBI 2025]

## Requirements
configargparse, logging, json, numpy, torch, torchvision, albumentations, opencv, tensorboardX

## Code Structure
```
mfcnet-tracker/
├── README.md
├── configs/          # Configuration files
├── models/           # Model definitions
├── scripts/          # Training and utility scripts
├── src/              # Source code
│   ├── dataloader.py
│   ├── engine.py
│   └── ...
└── utils/            # Utility functions
├── log_utils.py
└── ...
```

## Dataset 
The JIGSAWS annotated dataset is required for training and testing. It can be downloaded from [here](https://drive.google.com/drive/u/1/folders/1ln1X4V7cEV4q8lZLx_xN4FCvgs-rvpTY)
The expected dataset structure is:
```
dataset/
├── train/
│   ├── video_1/
│   │   ├── images/
│   │   │   └── frame<idx>.png
│   │   ├── pose_maps/
│   │   │   ├── framel<idx>.jpg (left tool annotation map)
│   │   │   └── framer<idx>.jpg (right tool annotation map)
│   │   └── depthmaps_depthanything_v2/
│   │       └── frame<idx>.png (depthmap outputs from DepthAnything-v2)
│   └── ... (other videos in the training set)
└── val/
    └── ... (similar structure to train, but for the validation set)
```
## Training 
Experiment results stored in RESULTS_DIR/RESULTS_NAME \
Ckpts stored in RESULTS_DIR/RESULTS_NAME/ckpts/ \
Logs (tensorboard and txt logs) in RESULTS_DIR/RESULTS_NAME/logs/ \
Outputs (if any) stored in RESULTS_DIR/RESULTS_NAME/outputs/

### Single-frame models
Type the following for more information. 
```
python scripts/train_toolpose_segmentation.py --help
```
Example command
```
python scripts/train_toolpose_segmentation.py \
--data_dir <JIGSAWS_ANNOTATED_DATASET_PATH> --dataset JIGSAWS --fold_index -1 \
--mode training --prediction toolpose_segmentation \
--expt_savedir <RESULTS_DIR> \
--expt_name <RESULTS_NAME> \
--print_freq 250 --save_freq 5 \
--lr 3e-5 --num_epochs 20 --scheduler StepDecay \
--batch_size 4 --num_workers 12 --num_classes 5 \
--metric_fns iou dice --loss_fns nll soft_jaccard --loss_wts 0.7 0.3 \
--class_weights 1 1000 1000 1000 1000 --seed 42 \
--model_type <MODEL_TYPE> --pretrained True \
--input_height 480 \
--input_width 640
```

### Multi-frame models
We train a MFCNet model on top of a pretrained single-frame (SFC) model. Type the following for more information. 
```
python scripts/train_multiframe_detection.py
```
Example command
```
python scripts/train_multiframe_detection.py \
--data_dir <JIGSAWS_ANNOTATED_DATASET_PATH> --dataset JIGSAWS --fold_index -1 \
--mode training --prediction_task toolpose_segmentation \
--num_input_frames 3 \
--expt_savedir <RESULTS_DIR> \
--expt_name <RESULTS_NAME> \
--print_freq 250 --save_freq 5 \
--lr 1e-4 --num_epochs 20 --scheduler StepDecay \
--batch_size 4 --num_workers 12 --num_classes 5 \
--metric_fns iou dice --loss_fns nll soft_jaccard --loss_wts 0.7 0.3 \
--class_weights 1 1000 1000 1000 1000 --seed 42 \
--model_type <MODEL_TYPE> --pretrained True \
--load_wts_base_model <PATH_TO_PRETRAINED_SINGLEFRAME_MODEL> \
--input_height 480 \
--input_width 640 \
--add_depth_inputs True \
--add_optflow_inputs True --train_base_model True
```

## Inference
Experiment results stored in RESULTS_DIR/RESULTS_NAME \
Metrics outputted to log in RESULTS_DIR/RESULTS_NAME/logs/ \
Output images (if any) stored in RESULTS_DIR/RESULTS_NAME/outputs/
### Single-frame models
Type the following for more information. 
```
python scripts/infer_toolpose_segmentation.py --help
```
Example command
```
python scripts/infer_toolpose_segmentation.py \
--data_dir <JIGSAWS_ANNOTATED_DATASET_PATH> --dataset JIGSAWS \
--prediction toolpose_segmentation \
--expt_savedir <RESULTS_DIR> \
--expt_name <RESULTS_NAME> \
--print_freq 250 --save_output_freq 5 \
--num_workers 12 --num_classes 5 \
--seed 42 \
--model_type <MODEL_TYPE> --pretrained True \
--input_height 480 \
--input_width 640 \
--load_wts_model <MODEL_TRAINED_WEIGHTS_PATH>
```
### Multi-frame models
Type the following for more information
```
python scripts/infer_multiframe_detection.py --help
```
Example command
```
python scripts/infer_multiframe_detection.py \
--data_dir <JIGSAWS_ANNOTATED_DATASET_PATH> --dataset JIGSAWS \
--prediction_task toolpose_segmentation \
--num_input_frames 3 \
--expt_savedir <RESULTS_DIR> \
--expt_name <RESULTS_NAME> \
--print_freq 250 --save_output_freq 2000 \
--num_workers 12 --num_classes 5 \
--seed 42 \
--model_type <MODEL_TYPE> --pretrained True \
--input_height 480 \
--input_width 640 \
--load_wts_base_model <BASE_MODEL_WEIGHTS_PATH> \
--load_wts_model <MODEL_TRAINED_WEIGHTS_PATH> \
--add_depth_inputs True \
--add_optflow_inputs True
```

## Testing on videos
Example commands
```
python scripts/test_toolpose_segmentation_on_videos_v2.py \
--videos_dir <JIGSAWS_VIDEO_DIR> \
--expt_savedir <RESULTS_DIR> \
--expt_name <RESULTS_NAME> \
--model_type <MODEL_TYPE> \
--num_input_frames 3 \
--load_wts_model <MODEL_TRAINED_WEIGHTS_PATH> \
--num_videos -1 --input_width 640 --input_height 480 \
--score_detection_threshold 0 --area_threshold 10 --dist_threshold 40 \
```
For multiframe, if using depth inputs too, you need to generate the monocular depth videos before hand. 
```
python scripts/test_multiframe_segmentation_on_videos_v3.py \
--videos_dir <JIGSAWS_VIDEO_DIR> \
--depth_videos_dir <JIGSAW_DEPTH_VIDEOS_DIR> \
--expt_savedir <RESULTS_DIR> \
--expt_name <RESULTS_NAME> \
--model_type <MODEL_TYPE> \
--num_input_frames 3 \
--load_wts_model <MODEL_TRAINED_WEIGHTS_PATH> \
--num_videos -1 --input_width 640 --input_height 480 \
--score_detection_threshold 0 --area_threshold 10 --dist_threshold 40 \
--add_optflow_inputs True \
--add_depth_inputs True
```
