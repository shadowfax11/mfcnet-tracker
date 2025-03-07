import numpy as np
import cv2 
import torch
from torchvision import transforms
import torchvision.transforms.functional as tF
from natsort import natsorted

def load_optflow_map(path, optflow_dir):
    with open(str(path).replace('images', optflow_dir).replace('jpg', 'flo')) as f:
        optflow = np.fromfile(f, dtype=np.float32)
        # optflow = optflow[2:].reshape((1024,1280,2))
        optflow = optflow[2:].reshape((480,640,2))
    return optflow

def load_attmap(file_name_list, idx, N): 
    if idx % N == 0: 
        # attmap = np.zeros((1024, 1280))
        attmap = np.zeros((480, 640))
    else:
        path = file_name_list[idx-1]
        attmap = cv2.imread(str(path).replace('images', 'attmaps').replace('jpg', 'png'),0)
    return attmap.astype(np.float32) / 255.0

def load_image(path):
    img = cv2.imread(str(path))
    if img is None: 
        print(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_depthmap(path):
    dmap = cv2.imread(str(path).replace('images','depth_maps_depthanythingv2').replace('jpg','png'))
    if dmap is None: 
        print(path)
    return cv2.cvtColor(dmap, cv2.COLOR_BGR2GRAY)

def load_mask(path, prediction_task):
    if prediction_task=='tooltip_segmentation':
        maskl = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framel').replace('jpg','png'))
        maskr = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framer').replace('jpg','png'))
        mask = np.zeros((maskl.shape[0], maskl.shape[1]))
        # only try to segment out the tool tips, ignore toolbase
        if np.amax(maskl):
            mask[np.where(maskl[:,:,0]>0)] = 255
            mask[np.where(maskl[:,:,2]>0)] = 255 
        if np.amax(maskr):
            mask[np.where(maskr[:,:,0]>0)] = 127
            mask[np.where(maskr[:,:,2]>0)] = 127
        return (mask / 127).astype(np.uint8)
    elif prediction_task=='endovis15_segmentation':
        maskl = cv2.imread(str(path).replace('images','pose_maps_endovis').replace('frame','framel').replace('jpg','png'))
        maskr = cv2.imread(str(path).replace('images','pose_maps_endovis').replace('frame','framer').replace('jpg','png'))
        mask = np.zeros((maskl.shape[0], maskl.shape[1]))
        if np.amax(maskl):
            mask[np.where(maskl[:,:,0]>0)] = 250
            mask[np.where(maskl[:,:,2]>0)] = 225 
            mask[np.where(maskl[:,:,1]==255)] = 200
            mask[np.where(maskl[:,:,1]==127)] = 175
            mask[np.where(maskl[:,:,1]==63)] = 150
        if np.amax(maskr):
            mask[np.where(maskr[:,:,0]>0)] = 125
            mask[np.where(maskr[:,:,2]>0)] = 100
            mask[np.where(maskr[:,:,1]==255)] = 75
            mask[np.where(maskr[:,:,1]==127)] = 50
            mask[np.where(maskr[:,:,1]==63)] = 25
        return (mask / 25).astype(np.uint8)
    elif prediction_task=='toolpose_segmentation':
        maskl = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framel').replace('jpg','png'))
        maskr = cv2.imread(str(path).replace('images','pose_maps').replace('frame','framer').replace('jpg','png'))
        mask = np.zeros((maskl.shape[0], maskl.shape[1]))
        if np.amax(maskl):
            mask[np.where(maskl[:,:,0]>0)] = 255
            mask[np.where(maskl[:,:,2]>0)] = 255 
            mask[np.where(maskl[:,:,1]>0)] = 191
        if np.amax(maskr):
            mask[np.where(maskr[:,:,0]>0)] = 127
            mask[np.where(maskr[:,:,2]>0)] = 127
            mask[np.where(maskr[:,:,1]>0)] = 63
        return (mask / 63).astype(np.uint8)
    elif prediction_task=='binary':
        binary_factor = 255
        mask_folder = 'binary_masks'
        mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)
        return (mask / binary_factor).astype(np.uint8)
    else:
        raise ValueError('Unknown prediction task: {}'.format(prediction_task))

def get_MICCAI2015_dataset_filenames(args): 
    if args.mode=='training': 
        folds = {-1: [], 0: [4], 1: [3], 2: [2], 3: [1]}
        train_path = args.data_dir / 'Tracking_Robotic_Training' / 'Training'
        train_file_names = [] 
        val_file_names = []
        for i in range(1,5): 
            train_file_names += natsorted(list((train_path / ('Dataset' + str(i)) / 'images').glob('*')), key=str)
        val_path = args.data_dir / 'Tracking_Robotic_Testing' / 'Tracking' 
        val_file_names = [] 
        for i in range(1,5): 
            val_file_names += natsorted(list((val_path / ('Dataset' + str(i)) / 'images').glob('*')), key=str)
        return train_file_names, val_file_names
    if args.mode=='testing':
        test_path = args.data_dir / 'Tracking_Robotic_Testing' / 'Tracking' 
        test_file_names = [] 
        for i in range(1,7): 
            test_file_names += natsorted(list((test_path / ('Dataset' + str(i)) / 'images').glob('*')), key=str)
        return test_file_names, None

def get_MICCAI2017_dataset_filenames(args):
    if args.mode=='training':
        folds = {-1: [], 0: [1, 3], 1: [2, 5],
                2: [4, 8], 3: [6, 7]}
        train_path = args.data_dir / 'cropped_train'
        train_file_names = []
        val_file_names = []
        for instrument_id in range(1, 9):
            if instrument_id in folds[args.fold_index]:
                val_file_names += natsorted(list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*')), key=str)
            else:
                train_file_names += natsorted(list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*')), key=str)
        return train_file_names, val_file_names
    if args.mode=='testing':
        test_path = args.data_dir / 'cropped_test'
        test_file_names = []
        for instrument_id in range(1, 11):
            test_file_names += natsorted(list((test_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*')), key=str)
        return test_file_names, None

def get_JIGSAWS_dataset_filenames(args):
    if args.mode=='training': 
        folds = {-1: [], 0: [1], 1: [2], 2: [1,2]}
        train_path = args.data_dir / 'annotations_train'
        val_path = args.data_dir / 'annotations_val'
        train_file_names = []
        val_file_names = [] 
        for i in range(1,7): 
            train_file_names += natsorted(list((train_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
            val_file_names += natsorted(list((val_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        # train_path = args.data_dir / 'train'
        # train_file_names = []
        # val_file_names = []
        # for i in range(1,7):
        #     if i<6:
        #         train_file_names += natsorted(list((train_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        #     else:
        #         val_file_names += natsorted(list((train_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        return train_file_names, val_file_names
    if args.mode=='testing':
        test_path = args.data_dir / 'annotations_val'
        test_file_names = []
        for i in range(1,7):
            test_file_names += natsorted(list((test_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        # test_path = args.data_dir / 'train' 
        # test_file_names = []
        # for i in range(1,7): 
        #     if i==6:
        #         test_file_names += natsorted(list((test_path / ('video_' + str(i)) / 'images').glob('*')), key=str)
        # for i in range(1,3): 
        #     test_file_names += natsorted(list((test_path / ('random_sample_set_' + str(i)) / 'images').glob('*')), key=str)
        return test_file_names, None


class to_tensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        return {'image': torch.from_numpy(image.transpose(2,0,1)), 
                'attmap': torch.from_numpy(attmap).unsqueeze(0), 
                'mask': torch.from_numpy(mask).unsqueeze(0)}

class customResize(object):
    def __call__(self, sample):
        image = sample['image'] 
        attmap = sample['attmap']
        mask = sample['mask']
        return {'image': transforms.Resize((480, 640))(image),
                'attmap': transforms.Resize((480, 640))(attmap),
                'mask': transforms.Resize((480, 640))(mask)}

class customRandomHSVDistortion(object):
    def __call__(self, sample):
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.2):
            image = tF.adjust_brightness(image, np.random.uniform(0.9,1.1))
            image = tF.adjust_contrast(image, np.random.uniform(0.9,1.1))
            image = tF.adjust_saturation(image, np.random.uniform(0.9,1.1))
        return {'image': image, 
                'attmap': attmap, 
                'mask': mask}

class customRandomRotation(object):
    def __call__(self, sample): 
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        angle = np.random.randint(-30,30)
        if np.random.binomial(size=1,n=1,p=0.2):
            return {'image': tF.rotate(image, angle),
                'attmap': tF.rotate(attmap, angle),
                'mask': tF.rotate(mask, angle)}
        else:
            return sample

class customVertFlip(object):
    """Flip the image vertically."""
    def __call__(self, sample):
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.5):
            return {'image': transforms.RandomVerticalFlip(p=0.5)(image), 
                'attmap': transforms.RandomVerticalFlip(p=0.5)(attmap),
                'mask': transforms.RandomVerticalFlip(p=0.5)(mask)}
        else: 
            return sample

class customHorzFlip(object):
    """Flip the image horizontally."""
    def __call__(self, sample): 
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.5):
            # mask[mask==1] = 3
            # mask[mask==2] = 1
            # mask[mask==3] = 2
            return {'image': transforms.RandomHorizontalFlip(p=0.5)(image), 
                'attmap': transforms.RandomHorizontalFlip(p=0.5)(attmap),
                'mask': transforms.RandomHorizontalFlip(p=0.5)(mask)}
        else: 
            return sample

class customHorzFlip_LR(object):
    """Flip the image horizontally."""
    def __call__(self, sample): 
        image = sample['image'] 
        attmap = sample['attmap'] 
        mask = sample['mask'] 
        if np.random.binomial(size=1,n=1,p=0.5):
            mask[mask==1] = 3
            mask[mask==2] = 1
            mask[mask==3] = 2
            return {'image': transforms.RandomHorizontalFlip(p=0.5)(image), 
                'attmap': transforms.RandomHorizontalFlip(p=0.5)(attmap),
                'mask': transforms.RandomHorizontalFlip(p=0.5)(mask)}
        else: 
            return sample

class customNormalize(object):
    """Normalize the image."""
    def __call__(self, sample):
        image = sample['image'].type(torch.float32)/255.0
        attmap = sample['attmap'].type(torch.float32)
        mask = sample['mask'] 
        return {'image': transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image), 
                'attmap': attmap,
                'mask': mask}    
