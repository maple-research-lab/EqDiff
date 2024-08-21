import torch
import torch.nn.functional as F
import torch.nn as nn

class AlignLossHook(nn.Module):
    @staticmethod
    def get_empty_store():
        return {
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }
    
    def __init__(self, loss_fn=nn.MSELoss(), loss_weights={'mid_self': 1.0, 'up_self': 1.0, 'down_self': 1.0}):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        
        self.num_att_layer = -1

        self.step_store = self.get_empty_store()
        self.loss_store = {}
    
    def reset(self):
        self.step_store = self.get_empty_store()
        self.loss_store = {}

    def __call__(self, sa, res_sa, place_in_unet='mid'):
        key = f'{place_in_unet}_self'
        loss = self.loss_fn(sa.float(), res_sa.float()) * self.loss_weights[key]
        #print(f'{key} of loss {loss} is appended')
        self.step_store[key].append(loss)

class MasksHook(nn.Module):
    @staticmethod
    def get_empty_store():
        return {
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }
    
    def __init__(self):
        super().__init__()
        self.num_att_layer = -1

        self.step_store = self.get_empty_store()
        self.mask_store = {}
    
    def reset(self):
        self.step_store = self.get_empty_store()
        self.mask_store = {}

    def __call__(self, mask, place_in_unet='mid'):
        key = f'{place_in_unet}_self'
        #print(f'{key} of loss {loss} is appended')
        self.step_store[key].append(mask)