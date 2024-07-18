from typing import Tuple, Union, Optional, List

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import numpy as np
from PIL import Image
from einops import rearrange

from diffusers import (
    AutoencoderKL
)

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from lamp.models.unet import UNet3DConditionModel

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]    
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


@torch.no_grad()
def get_text_embeddings(tokenizer, text_encoder, device, text: str) -> T:
    tokens = tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                   return_tensors="pt", return_overflowing_tokens=True).input_ids.to(device)
    return text_encoder(tokens).last_hidden_state.detach()

@torch.no_grad()
def denormalize(image, to_np=False):
    image = (image / 2 + 0.5).clamp(0, 1)
    if to_np:
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    else:
        image = image.cpu()
    return image


@torch.no_grad()
def decode(latent: T, vae: AutoencoderKL, im_cat: TN = None):
    image = vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image, to_np=False)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    #return Image.fromarray(image)
    return image

def init_pipe(device, dtype, unet_src, unet_tgt, scheduler) -> Tuple[UNet3DConditionModel, UNet3DConditionModel, T, T]:

    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    # for p in unet.parameters():
    #     p.requires_grad = False
    return unet_src, unet_tgt, alphas, sigmas


class DDSLoss:
    def __init__(self, device, 
                 unet_src: UNet3DConditionModel,
                 unet_tgt: UNet3DConditionModel,
                 scheduler: Union[
                    DDIMScheduler,
                    PNDMScheduler,
                    LMSDiscreteScheduler,
                    EulerDiscreteScheduler,
                    EulerAncestralDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                 ], 
                 dtype=torch.float32,
                 ):
        # self.t_min = 50
        # self.t_max = 600
        self.t_min = 50
        self.t_max = 900
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet_src, self.unet_tgt, self.alphas, self.sigmas = init_pipe(device, dtype, unet_src, unet_tgt, scheduler)
        self.prediction_type = scheduler.prediction_type

    # def noise_input(self, timesteps: torch.Tensor = None):
    #     alpha_t = [self.alphas[timestep, None, None, None] for timestep in timesteps]
    #     sigma_t = [self.sigmas[timestep, None, None, None] for timestep in timesteps]
    #     alpha_t = torch.cat(alpha_t, dim=0)
    #     sigma_t = torch.cat(sigma_t, dim=0)
    #     return alpha_t, sigma_t

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, text_embeddings_prior: T, alpha_t: T, sigma_t: T, get_raw=False, guidance_scale=7.5, is_src=False):
        bsz = z_t.shape[0]
        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = torch.cat([text_embeddings, text_embeddings_prior], 0).permute(1, 0, 2, 3).repeat_interleave(bsz//2, dim=1).reshape(-1, *text_embeddings.shape[2:])
        # embedd = text_embeddings.permute(1, 0, 2, 3).repeat_interleave(bsz, dim=1).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if is_src:
                latent_input = rearrange(latent_input, '(b f) c h w -> b c f h w', f=1).contiguous()
                print(f'latent_input: {latent_input.shape}, embedd: {embedd.shape}, timestep: {timestep.shape}')
                e_t_uc = self.unet_src(latent_input[:bsz], timestep[:bsz], embedd[:bsz]).sample
                e_t_c  = self.unet_src(latent_input[bsz:], timestep[bsz:], embedd[bsz:]).sample
                e_t = torch.cat([e_t_uc, e_t_c])
                e_t = rearrange(e_t, "b c f h w -> (b f) c h w", f=1).contiguous()
            else:
                latent_input = rearrange(latent_input, '(b f) c h w -> b c f h w', f=1).contiguous()
                e_t_uc = self.unet_tgt(latent_input[:bsz], timestep[:bsz], embedd[:bsz]).sample
                e_t_c  = self.unet_tgt(latent_input[bsz:], timestep[bsz:], embedd[bsz:]).sample
                e_t = torch.cat([e_t_uc, e_t_c])
                e_t = rearrange(e_t, "b c f h w -> (b f) c h w", f=1).contiguous()
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            if get_raw:
                return e_t_uncond, e_t
            print(f'e_t_uncond:{e_t_uncond.sum()}, e_t:{e_t.sum()}')
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        if get_raw:
            return e_t
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    def get_dds_loss(self, 
                     z_target: T, 
                     text_emb_source: T, 
                     text_emb_target: T,
                     text_emb_source_prior: T,
                     text_emb_target_prior: T,
                     eps=None, 
                     reduction='mean', 
                     symmetric: bool = False, 
                     calibration_grad=None, 
                     timesteps: torch.Tensor = None,
                     guidance_scale=7.5, 
                     raw_log=False,
                     mask=None
                     ) -> TS:
        with torch.inference_mode():
            z_t_target, eps, timesteps, alpha_t, sigma_t = self.noise_input(z_target)
            z_t_source = z_t_target #(b, c, h, w)
            print(f'z_t_source:{z_t_source.shape}, text_emb_source:{text_emb_source.shape}, timesteps:{timesteps.shape}, alpha_t:{alpha_t.shape}, sigma_t:{sigma_t.shape}')
            eps_pred_source, pred_z0_source = self.get_eps_prediction(
                                                  z_t_source,
                                                  timesteps,
                                                  text_emb_source,
                                                  text_emb_source_prior,
                                                  alpha_t,
                                                  sigma_t,
                                                  guidance_scale=guidance_scale,
                                                  is_src=True)
            eps_pred_target, pred_z0_target = self.get_eps_prediction(
                                                  z_t_target,
                                                  timesteps,
                                                  text_emb_target,
                                                  text_emb_target_prior,
                                                  alpha_t,
                                                  sigma_t,
                                                  guidance_scale=guidance_scale,
                                                  is_src=False)
            #eps_pred_source, eps_pred_target = eps_pred.chunk(2)
            # print(f'timestep:{timesteps}, grad_weights:{(alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp)}')
            grad = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp) * (eps_pred_source - eps_pred_target)
            if calibration_grad is not None:
                if calibration_grad.dim() == 4:
                    grad = grad - calibration_grad
                else:
                    grad = grad - calibration_grad[timesteps - self.t_min]
            if raw_log:
                log_loss = eps.detach().cpu(), eps_pred_target.detach().cpu(), eps_pred_source.detach().cpu()
            else:
                log_loss = (grad ** 2).mean()
        loss = z_target * grad.clone()
        if mask is not None:
            loss = ((loss * mask).sum([1,2,3])) / mask.sum([1,2,3]).mean()
        if symmetric:
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
            loss_symm = self.rescale * z_source * (-grad.clone())
            loss += loss_symm.sum() / (z_target.shape[2] * z_target.shape[3])
        elif reduction == 'mean':
            loss = loss.sum() / (z_target.shape[2] * z_target.shape[3])
        return loss, log_loss
