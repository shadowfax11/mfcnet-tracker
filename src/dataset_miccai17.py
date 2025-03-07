from torch.utils.data import Dataset 
import sys
sys.path.append('../utils/')
from utils.dataloader_utils import load_image, load_mask, load_depthmap

class MICCAI2017(Dataset):
    def __init__(self, file_names, transform, mode, prediction_task, num_input_frames, num_frames_per_video, add_depth_inputs=False): 
        self.file_names = file_names
        self.transform = transform
        self.mode = mode 
        self.prediction_task = prediction_task
        self.num_input_frames = num_input_frames
        self.num_frames_per_video = num_frames_per_video
        self.add_depth_inputs = add_depth_inputs
        self.N = len(self.file_names)

    def __len__(self): 
        return len(self.file_names)
    
    def __getitem__(self, idx): 
        img_file_name = self.file_names[idx]
        mask = load_mask(img_file_name, self.prediction_task)
        input = []
        input_depth = []
        last_valid_idx = -1
        for i in range(self.num_input_frames): 
            img_file_name = self.file_names[(idx - i)%self.N]
            if img_file_name.parent != self.file_names[idx].parent:
                img_file_name = self.file_names[last_valid_idx]
            else:
                last_valid_idx = (idx - i)%self.N
            input.append(load_image(img_file_name))
            if self.add_depth_inputs: 
                input_depth.append(load_depthmap(img_file_name))
        if self.add_depth_inputs:
            sample = {'input': input, 'mask': mask, 'input_depth': input_depth}
        else:
            sample = {'input': input, 'mask': mask}
        sample = self.transform(sample)
        return sample