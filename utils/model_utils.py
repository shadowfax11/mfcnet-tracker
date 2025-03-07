import math
import torch 
import os
from collections import OrderedDict

def save_model(model, ckpt_dir, optimizer=None, epoch=None): 
    torch.save({
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'epoch': epoch, 
        }, os.path.join(ckpt_dir, "model_{:03d}.pth".format(epoch)))
    return

def load_model_weights(model, loadpath, model_type):
    """
    Loads a checkpoint. Can load encoder-only, or full-model. 
    """
    if loadpath is not None:
        if not os.path.exists(loadpath):
            raise NameError("File {} does not exist".format(loadpath))
        state = torch.load(loadpath)
        epoch = state['epoch']
        if model_type=='DeepLab_v3' or model_type=='FCN':
            new_state_dict = OrderedDict()
            for key, value in state['model'].items():
                # Remove 'module.' prefix
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=False)
        else:
            new_state_dict = OrderedDict()
            for key, value in state['model'].items():
                # Remove 'module.' prefix
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
        return model, epoch, 1
    else: 
        return model, 1, 0
