import cv2 
import torch 
import numpy as np
from scipy import ndimage

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 as img_to_tensor
from albumentations import Compose, Resize, Normalize, VerticalFlip, RandomCrop, Rotate
from albumentations import GaussianBlur, MotionBlur, HueSaturationValue, ColorJitter
import sys
sys.path.append('./')
sys.path.append('../utils/')
from utils.dataloader_utils import load_image, load_mask, load_attmap, load_optflow_map
from utils.dataloader_utils import to_tensor, customHorzFlip_LR, customVertFlip, customNormalize, customResize, customRandomRotation, customRandomHSVDistortion
from utils.dataloader_utils import get_MICCAI2017_dataset_filenames, get_JIGSAWS_dataset_filenames, get_MICCAI2015_dataset_filenames

class RoboticSurgeryFramesDataset_withoptflow(Dataset):
    def __init__(self, file_names, optflow_dir, transform=None, 
                 mode='train', prediction_task='binary', 
                 num_frames=225):
        self.file_names = file_names
        self.transform = transform
        self.mode = mode
        self.prediction_task = prediction_task
        self.optflow_dir = optflow_dir
        self.N_frames = num_frames
        # initialize attmaps 
        self.init_attmaps()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.prediction_task)
        optflow = load_optflow_map(img_file_name, self.optflow_dir)
        attmap_prev = load_attmap(self.file_names, idx, self.N_frames)
        attmap = self.cal_attmap_np(attmap_prev, optflow)
        data = {'image': image, 'mask': mask, 'attmap': attmap}
        augmented = self.transform(data)
        image = torch.concat((augmented['image'], augmented['attmap']))
        mask = augmented['mask']
        return image, mask 
    
    def init_attmaps(self):
        for i in range(len(self.file_names)):
            img_file_name = self.file_names[i]
            # cv2.imwrite(str(img_file_name).replace('images', 'attmaps').replace('jpg', 'png'), np.zeros((1024, 1280), dtype=np.uint8))
            cv2.imwrite(str(img_file_name).replace('images', 'attmaps').replace('jpg', 'png'), np.zeros((480, 640), dtype=np.uint8))
        return 

    def cal_attmap_np(self, attmap_prev, optflow):
        '''
        Calculate Motion Flow based attention map
        input:
        attmap_prev: attention map of previous frame (stored in history)
        optflow: optical flow <prev_frame, cur_frame>
        return:
        attmap: Motion Flow based attention map for current frame
        '''
        h, w = optflow.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        new_x = np.rint(x + optflow[:,:,0]).astype(dtype=np.int64)
        new_y = np.rint(y + optflow[:,:,1]).astype(dtype=np.int64)
        # get valid x and valid y
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)
        attmap = np.zeros((h, w))
        attmap[new_y.flatten(), new_x.flatten()] = attmap_prev[y.flatten(), x.flatten()]
        # use the dilate operation to make attention area larger
        attmap = ndimage.grey_dilation(attmap, size=(10, 10))
        return attmap
    
class RoboticSurgeryFramesDataset(Dataset):
    def __init__(self, file_names, transform=None, mode='train', prediction_task='binary'):
        self.file_names = file_names
        self.transform = transform
        self.mode = mode
        self.prediction_task = prediction_task

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.prediction_task)
        data = {'image': image, 'mask': mask}
        augmented = self.transform(**data)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask 

# Custom vertical flip transformation class
class CustomVerticalFlip(A.DualTransform):
    def __init__(self, always_apply=False, p=0.5, task='tooltip_segmentation'):
        super(CustomVerticalFlip, self).__init__(always_apply, p)
        self.task = task

    def apply(self, img, **params):
        # Custom vertical flip for the image
        return np.ascontiguousarray(np.flip(img, axis=0))

    def apply_to_mask(self, mask, **params):
        # Custom vertical flip for the mask with remapping
        if self.task == 'endovis15_segmentation':
            mask[mask == 4] = 11
            mask[mask == 5] = 4
            mask[mask == 11] = 5
            mask[mask == 9] = 11
            mask[mask == 10] = 9
            mask[mask == 11] = 10
        return np.ascontiguousarray(np.flip(mask, axis=0))

    def get_transform_init_args_names(self):
        return ("always_apply", "p", "task")

# Custom horizontal flip transformation class
class CustomHorizontalFlip(A.DualTransform):
    def __init__(self, always_apply=False, p=0.5, task='tooltip_segmentation'):
        super(CustomHorizontalFlip, self).__init__(always_apply, p)
        self.task = task

    def apply(self, img, **params):
        # Custom horizontal flip for the image
        return np.ascontiguousarray(np.flip(img, axis=1))

    def apply_to_mask(self, mask, **params):
        # Custom horizontal flip for the mask with remapping
        if self.task == 'toolpose_segmentation':
            mask[mask == 1] = 5
            mask[mask == 3] = 1
            mask[mask == 5] = 3
            mask[mask == 2] = 5
            mask[mask == 4] = 2
            mask[mask == 5] = 4
        elif self.task == 'tooltip_segmentation':
            mask[mask == 1] = 3
            mask[mask == 2] = 1
            mask[mask == 3] = 2
        elif self.task == 'endovis15_segmentation':
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
        return np.ascontiguousarray(np.flip(mask, axis=1))

    def get_transform_init_args_names(self):
        return ("always_apply", "p", "task")

def get_transform(mode, args): 
    if args.add_optflow_inputs: 
            raise ValueError('Not implemented yet')
    if mode=='train': 
        transform = Compose([Resize(args.input_height, args.input_width), CustomVerticalFlip(p=0.5, task=args.prediction_task), #VerticalFlip(p=0.5), 
                             CustomHorizontalFlip(p=0.5, task=args.prediction_task), 
                             Rotate(limit=(-15,15), p=1), ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0, p=0.5),
                            # HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=0, p=0.5),
                            Normalize(p=1), 
                            img_to_tensor()])
    elif mode=='val':
        transform = Compose([Resize(args.input_height, args.input_width), 
                             Normalize(p=1), 
                             img_to_tensor()])
    elif mode=='test':
        transform = Compose([Resize(args.input_height, args.input_width), 
                             Normalize(p=1),  
                             img_to_tensor()]) 
    else:
        raise NotImplementedError
    return transform

def get_data_loader(args):
    if args.dataset == 'MICCAI2017': 
        if args.mode=='training':
            train_file_names, val_file_names = get_MICCAI2017_dataset_filenames(args)
            if not args.add_optflow_inputs:
                train_transform = get_transform('train', args)
                val_transform = get_transform('val', args)
                train_dataset = RoboticSurgeryFramesDataset(train_file_names, transform=train_transform, mode=args.mode, prediction_task=args.prediction_task)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
                val_dataset = RoboticSurgeryFramesDataset(val_file_names, transform=val_transform, mode=args.mode, prediction_task=args.prediction_task)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else: 
                assert 'TAPNet' in args.model_type
                train_transform = get_transform('train', args)
                val_transform = get_transform('val', args)
                # train_transform = transforms.Compose([to_tensor(), customHorzFlip_LR(), customVertFlip(), customNormalize()])
                # val_transform = transforms.Compose([to_tensor(), customNormalize()])
                train_dataset = RoboticSurgeryFramesDataset_withoptflow(train_file_names, args.optflow_dir, transform=train_transform, mode=args.mode, prediction_task=args.prediction_task, num_frames=args.num_frames_per_video)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                val_dataset = RoboticSurgeryFramesDataset_withoptflow(val_file_names, args.optflow_dir, transform=val_transform, mode=args.mode, prediction_task=args.prediction_task, num_frames=args.num_frames_per_video)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return train_loader, val_loader
        if args.mode=='testing':
            test_file_names, _ = get_MICCAI2017_dataset_filenames(args)
            if not args.add_optflow_inputs:
                test_transform = get_transform('test', args)
                test_dataset = RoboticSurgeryFramesDataset(test_file_names, transform=test_transform, mode=args.mode, prediction_task=args.prediction_task)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else: 
                assert 'TAPNet' in args.model_type
                test_transform = get_transform('test', args)
                # test_transform = transforms.Compose([to_tensor(), customNormalize()])
                test_dataset = RoboticSurgeryFramesDataset_withoptflow(test_file_names, args.optflow_dir, transform=test_transform, mode=args.mode, prediction_task=args.prediction_task, num_frames=args.num_frames_per_video)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return None, test_loader
    if args.dataset == 'MICCAI2015': 
        assert args.prediction_task == 'tooltip_segmentation' or args.prediction_task == 'toolpose_segmentation' or args.prediction_task == 'endovis15_segmentation'
        if args.mode=='training': 
            train_file_names, val_file_names = get_MICCAI2015_dataset_filenames(args)
            if not args.add_optflow_inputs:
                train_transform = get_transform('train', args)
                val_transform = get_transform('val', args)
                train_dataset = RoboticSurgeryFramesDataset(train_file_names, transform=train_transform, mode=args.mode, prediction_task=args.prediction_task)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
                val_dataset = RoboticSurgeryFramesDataset(val_file_names, transform=val_transform, mode=args.mode, prediction_task=args.prediction_task)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else:
                raise NotImplementedError
            return train_loader, val_loader
        if args.mode=='testing':
            test_file_names, _ = get_MICCAI2015_dataset_filenames(args)
            if not args.add_optflow_inputs:
                test_transform = get_transform('test', args)
                test_dataset = RoboticSurgeryFramesDataset(test_file_names, transform=test_transform, mode=args.mode, prediction_task=args.prediction_task)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else:
                raise NotImplementedError
            return None, test_loader
    if args.dataset == 'JIGSAWS': 
        if args.mode=='training':
            train_file_names, val_file_names = get_JIGSAWS_dataset_filenames(args)
            if not args.add_optflow_inputs:
                train_transform = get_transform('train', args)
                val_transform = get_transform('val', args)
                train_dataset = RoboticSurgeryFramesDataset(train_file_names, transform=train_transform, mode=args.mode, prediction_task=args.prediction_task)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
                val_dataset = RoboticSurgeryFramesDataset(val_file_names, transform=val_transform, mode=args.mode, prediction_task=args.prediction_task)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else: 
                assert 'TAPNet' in args.model_type
                train_transform = get_transform('train', args)
                val_transform = get_transform('val', args)
                # train_transform = transforms.Compose([to_tensor(), customResize(), customVertFlip(), customRandomRotation(), customRandomHSVDistortion(), customNormalize()])
                # val_transform = transforms.Compose([to_tensor(), customResize(), customNormalize()])
                train_dataset = RoboticSurgeryFramesDataset_withoptflow(train_file_names, args.optflow_dir, transform=train_transform, mode=args.mode, prediction_task=args.prediction_task, num_frames=args.num_frames_per_video)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                val_dataset = RoboticSurgeryFramesDataset_withoptflow(val_file_names, args.optflow_dir, transform=val_transform, mode=args.mode, prediction_task=args.prediction_task, num_frames=args.num_frames_per_video)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return train_loader, val_loader
        if args.mode=='testing':
            test_file_names, _ = get_JIGSAWS_dataset_filenames(args)
            if not args.add_optflow_inputs:
                test_transform = get_transform('test', args)
                test_dataset = RoboticSurgeryFramesDataset(test_file_names, transform=test_transform, mode=args.mode, prediction_task=args.prediction_task)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            else: 
                assert 'TAPNet' in args.model_type
                test_transform = get_transform('test', args)
                # test_transform = transforms.Compose([to_tensor(), customNormalize()])
                test_dataset = RoboticSurgeryFramesDataset_withoptflow(test_file_names, args.optflow_dir, transform=test_transform, mode=args.mode, prediction_task=args.prediction_task, num_frames=args.num_frames_per_video)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            return None, test_loader
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

if __name__ == '__main__':
    from types import SimpleNamespace
    from pathlib import Path 
    import matplotlib.pyplot as plt 

    args = SimpleNamespace(
        data_dir = Path('/shared/bg40/surgical_video_datasets/miccai2015'), 
        dataset = 'MICCAI2015', 
        input_height = 576,
        input_width = 720,
        mode = 'training',
        prediction_task = 'endovis15_segmentation',
        batch_size = 1, 
        num_workers = 1,
        add_depth_inputs = False, 
        add_optflow_inputs = False
    )

    train_loader, val_loader = get_data_loader(args)
    print("Train loader length: ", len(train_loader))
    print("Val loader length: ", len(val_loader))
    for i, sample in enumerate(train_loader):
        image, mask = sample
        print("Image shape: ", image.shape)
        print("Mask shape: ", mask.shape)
        plt.figure(figsize=(10, 5), dpi=300)  # Increase the figure size and DPI for sharpness
        # Plot the image
        plt.subplot(121)
        plt.imshow(image[0].permute(1, 2, 0).numpy() + 0.5)
        plt.axis('off')  # Remove axis for a cleaner look
        plt.title('Image', fontsize=14)  # Increase font size for readability
        # Plot the mask
        mask = mask[0].numpy()
        # mask_vis = mask * 60
        mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
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
        plt.subplot(122)
        plt.imshow(mask_vis.astype(np.uint8))
        plt.axis('off')
        plt.title('Mask', fontsize=14)
        # Save the figure with reduced margins
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)  # Remove excess padding
        plt.savefig('sample_{:03d}.png'.format(i), bbox_inches='tight')  # Remove excess white space
        if i==10:
            break
