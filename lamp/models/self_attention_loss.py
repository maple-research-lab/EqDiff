import os
import os.path as osp
import random
import shutil
from pathlib import Path
from einops import rearrange, repeat
from PIL import ImageFilter
import torchvision.transforms.functional as TF
import torchvision.utils as T
import torchvision
from typing import List, Optional


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint



class SALoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
                attention_maps, #(h,w,num_pixels_ref)
                same_ref_flags, #(num_ref)
                mask_gt, #(h, w)
                mask_ref,#(num_ref, h, w)
                reduction='mean',
                step=None,
                output_dir=None,
                enable_log=False): 
        # print(f'attention_maps:{attention_maps.shape}, same_ref_flags:{same_ref_flags.shape}, mask_gt:{mask_gt.shape}, mask_ref:{mask_ref.shape}')
        h, w, l = attention_maps.shape
        num_ref = int(l//h//w)
        assert num_ref == same_ref_flags.shape[0]

        # print(h, w, num_ref)
        attention_maps = attention_maps.reshape(h, w, num_ref, h, w)

        #attention_maps = rearrange(attention_maps, "l n a b -> l a (n b)")

        # print(f'mask_ref before interpolate:{mask_ref}')
        mask_ref = F.interpolate(mask_ref.unsqueeze(0), (h, w), mode='nearest')[0]
        # print(f'mask_ref after interpolate:{mask_ref}')
        mask_ref_same = same_ref_flags[..., None, None] * mask_ref #(num_ref, h, w)
        # print(f'mask_ref_same:{mask_ref_same}')
        # filter out the different concept attention scores
        # print(f'attention maps before:{attention_maps.sum([2,3,4])}')
        
        attention_maps = attention_maps * mask_ref_same[None, None, ...] #(h, w, num_ref, h, w)
        # print(f'attention maps after:{attention_maps.sum([2,3,4])}')
        attention_maps = rearrange(attention_maps, 'h w n hr wr -> h w (n hr wr)')
        attention_maps = attention_maps.sum([-1]) #(h,w)

        # calculate the loss
        mask_gt = F.interpolate(mask_gt.unsqueeze(0), (h, w), mode='nearest')[0][0]#(h,w)
        loss = F.mse_loss(attention_maps*mask_gt, mask_gt, reduce=reduction)

        if output_dir is not None and enable_log:
            save_dir = osp.join(output_dir, 'sa_loss')
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(attention_maps.unsqueeze(0).unsqueeze(0), osp.join(save_dir, f'sa_sum_{step:05}.jpg'))
            torchvision.utils.save_image(mask_gt.unsqueeze(0).unsqueeze(0), osp.join(save_dir, f'sa_gt_{step:05}.jpg'))

        #loss = (attention_maps - mask_gt)
        return loss
        