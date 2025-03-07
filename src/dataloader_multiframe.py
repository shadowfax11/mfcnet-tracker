import cv2 
import torch 
import numpy as np 
import sys; sys.path.append('./')

from src.dataset_miccai2015 import MICCAI2015
from src.dataset_miccai17 import MICCAI2017
from src.dataset_jigsaws import JIGSAWS
from torch.utils.data import DataLoader 
from torchvision import transforms
import torchvision.transforms.functional as tF

from utils.dataloader_utils import get_MICCAI2015_dataset_filenames, get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames

class to_tensor(object): 
    def __call__(self, sample): 
        input = sample['input'] 
        mask = sample['mask'] 
        tensor_dict = {} 
        tensor_dict['input'] = []
        tensor_dict['mask'] = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        for image in input: 
            tensor_dict['input'].append(torch.from_numpy(image.transpose(2,0,1).astype(np.float32)/255.0))
        if 'input_depth' in sample: 
            input_depth = sample['input_depth']
            tensor_dict['input_depth'] = []
            for depth in input_depth: 
                tensor_dict['input_depth'].append(torch.from_numpy(depth.astype(np.float32)/255.0).unsqueeze(0))
        return tensor_dict
    
class customResize(object):
    def __init__(self, img_size):
        if isinstance(img_size, int): 
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            assert len(img_size) == 2
            self.img_size = img_size
        else:
            raise TypeError
        self.img_size = img_size 
    
    def __call__(self, sample): 
        input = sample['input'] 
        mask = sample['mask'] 
        resized_dict = {} 
        resized_dict['input'] = []
        resized_dict['mask'] = transforms.Resize(self.img_size, interpolation=tF.InterpolationMode.NEAREST)(mask)
        for image in input: 
            resized_dict['input'].append(transforms.Resize(self.img_size, interpolation=tF.InterpolationMode.BILINEAR)(image))
        if 'input_depth' in sample:
            input_depth = sample['input_depth']
            resized_dict['input_depth'] = []
            for depth in input_depth: 
                resized_dict['input_depth'].append(transforms.Resize(self.img_size, interpolation=tF.InterpolationMode.NEAREST)(depth))
        return resized_dict

class customRandomRotate(object): 
    def __call__(self, sample): 
        angle = np.random.randint(-15, 15) #(-30, 30)
        input = sample['input']
        mask = sample['mask']
        rotated_dict = {}
        rotated_dict['input'] = []
        rotated_dict['mask'] = tF.rotate(mask, angle)
        for image in input:
            rotated_dict['input'].append(tF.rotate(image, angle))
        if 'input_depth' in sample:
            input_depth = sample['input_depth']
            rotated_dict['input_depth'] = []
            for depth in input_depth:
                rotated_dict['input_depth'].append(tF.rotate(depth, angle))
        return rotated_dict

class customRandomHSVDistortion(object): 
    def __init__(self, p=0.5): 
        self.p = p 
    
    def __call__(self, sample): 
        input = sample['input'] 
        mask = sample['mask'] 
        distorted_dict = {}
        distorted_dict['input'] = []
        distorted_dict['mask'] = mask
        if np.random.binomial(size=1, n=1, p=self.p): 
            for image in input: 
                image = tF.adjust_brightness(image, np.random.uniform(0.95,1.05))
                image = tF.adjust_contrast(image, np.random.uniform(0.95,1.05))
                image = tF.adjust_saturation(image, np.random.uniform(0.95,1.05))
                distorted_dict['input'].append(image)
        else: 
            distorted_dict['input'] = input
        if 'input_depth' in sample: 
            input_depth = sample['input_depth']
            distorted_dict['input_depth'] = input_depth
        return distorted_dict

class customHorizontalFlip(object): 
    def __init__(self, prediction_task, p=0.5): 
        self.task = prediction_task
        self.p = p
    
    def __call__(self, sample): 
        input = sample['input']
        mask = sample['mask']
        flipped_dict = {}
        flipped_dict['input'] = []
        if np.random.binomial(size=1, n=1, p=self.p):
            if 'input_depth' in sample:
                flipped_dict['input_depth'] = []
                for depth in sample['input_depth']: 
                    flipped_dict['input_depth'].append(transforms.RandomHorizontalFlip(p=1)(depth))
            if self.task == 'binary': 
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input: 
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
            if self.task == 'tooltip_segmentation':
                mask[mask==1] = 3
                mask[mask==2] = 1
                mask[mask==3] = 2
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input: 
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
            if self.task == 'toolpose_segmentation':
                mask[mask==1] = 5
                mask[mask==3] = 1
                mask[mask==5] = 3
                mask[mask==2] = 5
                mask[mask==4] = 2
                mask[mask==5] = 4
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input:
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
            if self.task == 'endovis15_segmentation':
                mask[mask == 1] = 11
                mask[mask == 6] = 1
                mask[mask == 11] = 6
                mask[mask == 2] = 11
                mask[mask == 7] = 2
                mask[mask == 11] = 7
                mask[mask == 3] = 11
                mask[mask == 8] = 3
                mask[mask == 11] = 8
                mask[mask == 4] = 11
                mask[mask == 10] = 4
                mask[mask == 11] = 10
                mask[mask == 5] = 11
                mask[mask == 9] = 5
                mask[mask == 11] = 9
                flipped_dict['mask'] = transforms.RandomHorizontalFlip(p=1)(mask)
                for image in input:
                    flipped_dict['input'].append(transforms.RandomHorizontalFlip(p=1)(image))
                return flipped_dict
        else:
            return sample

class customVerticalFlip(object):
    def __init__(self, prediction_task, p=0.5): 
        self.task = prediction_task
        self.p = p

    def __call__(self, sample): 
        input = sample['input'] 
        mask = sample['mask'] 
        flipped_dict = {} 
        flipped_dict['input'] = []
        if np.random.binomial(size=1, n=1, p=self.p):
            if self.task == 'endovis15_segmentation':
                mask[mask == 4] = 11
                mask[mask == 5] = 4
                mask[mask == 11] = 5
                mask[mask == 9] = 11
                mask[mask == 10] = 9
                mask[mask == 11] = 10
            flipped_dict['mask'] = transforms.RandomVerticalFlip(p=1)(mask)
            if 'input_depth' in sample:
                flipped_dict['input_depth'] = []
                for depth in sample['input_depth']:
                    flipped_dict['input_depth'].append(transforms.RandomVerticalFlip(p=1)(depth))
            for image in input: 
                flipped_dict['input'].append(transforms.RandomVerticalFlip(p=1)(image))
            return flipped_dict
        else:
            return sample

class customNormalize(object): 
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]): 
        self.mean = mean 
        self.std = std 
    
    def __call__(self, sample): 
        input = sample['input'] 
        mask = sample['mask'] 
        normalized_dict = {} 
        normalized_dict['input'] = []
        normalized_dict['mask'] = mask
        for image in input: 
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            normalized_dict['input'].append(image)
        if 'input_depth' in sample:
            input_depth = sample['input_depth']
            normalized_dict['input_depth'] = []
            for depth in input_depth: 
                normalized_dict['input_depth'].append(depth)
        return normalized_dict

def get_transform(mode, args):
    if mode == 'train': 
        transform_list = [to_tensor(), customRandomRotate(), customRandomHSVDistortion(), 
                          customResize((args.input_height, args.input_width)), 
                          customVerticalFlip(args.prediction_task), customHorizontalFlip(args.prediction_task), 
                          customNormalize()]
    elif mode == 'val': 
        transform_list = [to_tensor(), customResize((args.input_height, args.input_width)), 
                          customNormalize()]
    elif mode == 'test': 
        transform_list = [to_tensor(), customResize((args.input_height, args.input_width)), 
                          customNormalize()]
    else: 
        raise NotImplementedError
    return transforms.Compose(transform_list)

def get_data_loader(args): 
    if args.dataset == 'MICCAI2017': 
        if args.mode == 'training': 
            train_file_names, val_file_names = get_MICCAI2017_dataset_filenames(args) 
            train_transform = get_transform('train', args)
            val_transform = get_transform('val', args)
            train_dataset = MICCAI2017(train_file_names, train_transform, 
                                       mode=args.mode, prediction_task=args.prediction_task, 
                                       num_input_frames=args.num_input_frames, 
                                       num_frames_per_video=args.num_frames_per_video)
            val_dataset = MICCAI2017(val_file_names, val_transform, 
                                     mode=args.mode, prediction_task=args.prediction_task,
                                     num_input_frames=args.num_input_frames,
                                     num_frames_per_video=args.num_frames_per_video)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                      shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True)
            return train_loader, val_loader
        else: 
            test_file_names, _ = get_MICCAI2017_dataset_filenames(args)
            test_transform = get_transform('test', args)
            test_dataset = MICCAI2017(test_file_names, test_transform, 
                                      mode=args.mode, prediction_task=args.prediction_task,
                                     num_input_frames=args.num_input_frames, 
                                     num_frames_per_video=args.num_frames_per_video)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True)
            return None, test_loader
    elif args.dataset == 'MICCAI2015':
        if args.mode == 'training': 
            train_file_names, val_file_names = get_MICCAI2015_dataset_filenames(args)
            train_transform = get_transform('train', args)
            val_transform = get_transform('val', args)
            train_dataset = MICCAI2015(train_file_names, train_transform,
                                    mode=args.mode, prediction_task=args.prediction_task,
                                    num_input_frames=args.num_input_frames,
                                    num_frames_per_video=args.num_frames_per_video, 
                                    add_depth_inputs=args.add_depth_inputs)
            val_dataset = MICCAI2015(val_file_names, val_transform,
                                    mode=args.mode, prediction_task=args.prediction_task,
                                    num_input_frames=args.num_input_frames,
                                    num_frames_per_video=76, 
                                    add_depth_inputs=args.add_depth_inputs)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            return train_loader, val_loader
        if args.mode == 'testing': 
            test_file_names, _ = get_MICCAI2015_dataset_filenames(args)
            test_transform = get_transform('test', args)
            test_dataset = MICCAI2015(test_file_names, test_transform, 
                                      mode=args.mode, prediction_task=args.prediction_task,
                                     num_input_frames=args.num_input_frames, 
                                     num_frames_per_video=args.num_frames_per_video, 
                                     add_depth_inputs=args.add_depth_inputs)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True)
            return None, test_loader
    elif args.dataset == 'JIGSAWS': 
        if args.mode == 'training':
            train_file_names, val_file_names = get_JIGSAWS_dataset_filenames(args)
            train_transform = get_transform('train', args)
            val_transform = get_transform('val', args)
            train_dataset = JIGSAWS(train_file_names, train_transform,
                                    mode=args.mode, prediction_task=args.prediction_task,
                                    num_input_frames=args.num_input_frames,
                                    num_frames_per_video=args.num_frames_per_video, 
                                    add_depth_inputs=args.add_depth_inputs)
            val_dataset = JIGSAWS(val_file_names, val_transform,
                                    mode=args.mode, prediction_task=args.prediction_task,
                                    num_input_frames=args.num_input_frames,
                                    num_frames_per_video=args.num_frames_per_video, 
                                    add_depth_inputs=args.add_depth_inputs)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            return train_loader, val_loader
        else:
            test_file_names, _ = get_JIGSAWS_dataset_filenames(args)
            test_transform = get_transform('test', args)
            test_dataset = JIGSAWS(test_file_names, test_transform, 
                                      mode=args.mode, prediction_task=args.prediction_task,
                                     num_input_frames=args.num_input_frames, 
                                     num_frames_per_video=args.num_frames_per_video, 
                                     add_depth_inputs=args.add_depth_inputs)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True)
            return None, test_loader
    else: 
        raise NotImplementedError

if __name__=="__main__":
    from types import SimpleNamespace
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Create a SimpleNamespace object
    args = SimpleNamespace(
        data_dir=Path('/shared/bg40/surgical_video_datasets/miccai2015/'),
        dataset='MICCAI2015', 
        input_height=480,
        input_width=640,
        mode='training',
        prediction_task='endovis15_segmentation',
        num_input_frames=3,
        num_frames_per_video=225,
        batch_size=1,
        num_workers=1,
        add_depth_inputs=False
        )

    train_loader, val_loader = get_data_loader(args)
    print("Train loader length: ", len(train_loader))
    print("Val loader length: ", len(val_loader))
    for i, sample in enumerate(train_loader): 
        print("Sample keys: ", sample.keys())
        plt.figure(figsize=(12, 4), dpi=300)  # Increase DPI for sharper images
        # Plot the input frames
        plt.subplot(2, 3, 1)
        plt.imshow(sample['input'][0][0].permute(1, 2, 0).numpy() + 0.5)
        plt.title("Input frame 1", fontsize=14)  # Increased font size
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(sample['input'][1][0].permute(1, 2, 0).numpy() + 0.5)
        plt.title("Input frame 2", fontsize=14)
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(sample['input'][2][0].permute(1, 2, 0).numpy() + 0.5)
        plt.title("Input frame 3", fontsize=14)
        plt.axis('off')
        # Plot the mask
        plt.subplot(2, 3, 4)
        # plt.imshow(sample['mask'][0][0].numpy() * 63, cmap='gray')  # Assuming a grayscale mask
        mask = sample['mask'][0][0].numpy()
        mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_vis[mask == 1] = [0, 63, 0]
        mask_vis[mask == 2] = [0, 127, 0]
        mask_vis[mask == 3] = [0, 255, 0]
        mask_vis[mask == 4] = [255, 0, 0]
        mask_vis[mask == 5] = [0, 0, 255]
        mask_vis[mask == 6] = [0, 63, 0]
        mask_vis[mask == 7] = [0, 127, 0]
        mask_vis[mask == 8] = [0, 255, 0]
        mask_vis[mask == 9] = [255, 0, 0]
        mask_vis[mask == 10] = [0, 0, 255]
        plt.imshow(mask_vis)
        plt.title("Mask", fontsize=14)
        plt.axis('off')
        # Adjust layout and save with reduced margins
        plt.tight_layout(pad=0, h_pad=0.5, w_pad=0.5)  # Reduce padding between subplots
        plt.savefig('sample_{:03d}.png'.format(i), bbox_inches='tight')  # Save without excess white space
        plt.close()  # Close the figure to free memory
        if i==5: 
            break

