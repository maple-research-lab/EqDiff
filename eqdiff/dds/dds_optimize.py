import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import os.path as osp
import random
import shutil
import warnings
from pathlib import Path
from einops import rearrange, repeat
from PIL import ImageFilter
import torchvision.transforms.functional as TF
import torchvision.utils as T
import torchvision
from typing import List, Optional


import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTokenizer
from collections import defaultdict
from omegaconf import OmegaConf

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from eqdiff.dds.dds import *

def DDS_optimize(accelerator: Accelerator,
                 unet_tgt: UNet3DConditionModel, 
                 unet_src: UNet3DConditionModel, 
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 scheduler: DDIMScheduler,
                 vae: AutoencoderKL,
                 text_source: str, 
                 text_source_prior: str,
                 text_target: str, 
                 text_target_prior: str,
                 latents: torch.Tensor,
                 mask: torch.Tensor,
                 timesteps: torch.Tensor,
                 eps: torch.Tensor,
                 lambda_ddsloss: float, 
                 lambda_cusloss: float,
                 device: torch.device,
                 num_iters=200, 
                 optimizer=None,
                 lr_scheduler=None,
                 global_step=0,
                 max_grad_norm=1.0,
                 guidance_scale=7.5,
                 save_image=False,
                 ):
    # freeze unet_tgt's cross attention k,v as it sync with unet_src's update

    #
    dds_loss = DDSLoss(device, unet_src, unet_tgt, scheduler)

    with torch.no_grad():
        embedding_null = get_text_embeddings(tokenizer, text_encoder, device, "")
        embedding_text = get_text_embeddings(tokenizer, text_encoder, device, text_source)
        embedding_text_target = get_text_embeddings(tokenizer, text_encoder, device, text_target)#(1,77,768)
        embedding_source = torch.stack([embedding_null, embedding_text], dim=1)#(1, 2, 77,768)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)#(1,2, 77,768)

        embedding_text_prior = get_text_embeddings(tokenizer, text_encoder, device, text_source_prior)#(1,77,768)
        embedding_text_target_prior = get_text_embeddings(tokenizer, text_encoder, device, text_target_prior)#(1,77,768)
        embedding_source_prior = torch.stack([embedding_null, embedding_text_prior], dim=1)#(1,2, 77,768)
        embedding_target_prior = torch.stack([embedding_null, embedding_text_target_prior], dim=1)#(1,2, 77,768)
    
    # param_optimize = [unet_tgt.parameters()]
    # optimizer = SGD(params=itertools.chain(*param_optimize), lr=dds_lr)

    bsz = latents.shape[0] # ori+prior
    # embedding_text_target = embedding_text_target.repeat(bsz, 1, 1)#(b,77,768)
    embedding_text_target = torch.cat([embedding_text_target, embedding_text_target_prior], 0).repeat_interleave(bsz//2, 0)#(b,77,768)

    for i in tqdm(range(num_iters)):
        
        # 1. prepare input for optimization
        noisy_latents, eps, timesteps, alpha_t, sigma_t = dds_loss.noise_input(latents)

        noisy_latents_reshape = rearrange(noisy_latents, '(b f) c h w -> b c f h w', f=1).contiguous()
        z_target = unet_tgt(noisy_latents_reshape, timesteps, embedding_text_target).sample # with reference attention
        z_target = rearrange(z_target, 'b c f h w -> (b f) c h w').contiguous()

        #print(f'z_target:{z_target.shape}, noisy_latents:{noisy_latents.shape}')
        pred_z0_target = (noisy_latents - z_target * sigma_t) / alpha_t

        # os.makedirs('./tmp/dds_log', exist_ok=True)
        # torchvision.utils.save_image(decode(pred_z0_target, vae), f'./tmp/dds_log/pred_z0_target_{global_step:04}_{i:04}.png')

        # 4. run optimization
        #print(f'pred_z0_target: {pred_z0_target.shape}, embedding_source: {embedding_source.shape}, embedding_target: {embedding_target.shape}')
        loss = 0
        loss_dds, log_loss = dds_loss.get_dds_loss(pred_z0_target, embedding_source, embedding_target, \
                                                   embedding_source_prior, embedding_target_prior, \
                                                   guidance_scale=guidance_scale, \
                                                   raw_log=False, mask=mask)
        loss += loss_dds * lambda_ddsloss
        
        # 5. add the customization loss
        if mask is not None:
            loss_custom = F.mse_loss(z_target.float(), eps.float(), reduction="none")
            loss_custom = ((loss_custom * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
        else:
            loss_custom = F.mse_loss(z_target.float(), eps.float(), reduction="mean")
        loss += loss_custom * lambda_cusloss
        
        print(f'iteration {i}: loss:{loss.detach().item()}, loss_dds:{loss_dds.detach().item()}, loss_custom:{loss_custom.detach().item()}')
        # print(f'iteration {i}: loss:{loss.detach().item()}, loss_dds:{loss_dds.detach().item()}, loss_custom:{loss_custom.detach().item()}')
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(itertools.chain(unet_tgt.parameters()), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
  