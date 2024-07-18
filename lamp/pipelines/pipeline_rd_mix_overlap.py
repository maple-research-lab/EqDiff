# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import cv2
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision

from math import sqrt
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from lamp.util import save_videos_grid, ddim_inversion

from lamp.models.controlnet import ControlNetModel
from torch.optim.sgd import SGD

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin

from einops import rearrange

from ..models.unet import UNet3DConditionModel

from lamp.models.mutual_self_attention import ReferenceAttentionControl
from lamp.models.reference_encoder import AppearanceEncoderModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def calc_mean_std(feat, eps=1e-8):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

#AadIN
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

@dataclass
class LAMPPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    image: Union[torch.Tensor, np.ndarray]
    ref_image_tensor: torch.Tensor


class LAMPPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet_phase2: UNet3DConditionModel, # unet trained in a early stage with text embedding and k/v 
        unet_src: UNet3DConditionModel, # unet trained all the time without reference encoder
        unet: UNet3DConditionModel, # unet with reference encoder
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        dds_scheduler: Optional[DDPMScheduler],
        reference_encoder: Union[AppearanceEncoderModel, None],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            unet_phase2=unet_phase2,
            unet_src=unet_src,
            reference_encoder=reference_encoder,
            scheduler=scheduler,
            dds_scheduler=dds_scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(self, condition, num_videos_per_prompt, device, dtype, do_classifier_free_guidance):
        # prepare conditions for controlnet
        condition = torch.from_numpy(condition.copy()).to(device=device, dtype=dtype) / 255.0
        condition = torch.stack([condition for _ in range(num_videos_per_prompt)], dim=0)
        condition = rearrange(condition, 'b f h w c -> (b f) c h w').clone()
        if do_classifier_free_guidance:
            condition = torch.cat([condition] * 2)
        return condition

    @torch.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "f h w c -> f c h w").to(device)
        latents = []
        for frame_idx in range(images.shape[0]):
            latents.append(self.vae.encode(images[frame_idx:frame_idx+1])['latent_dist'].mean * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def images2tensor(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "f h w c -> f c h w").to(device)

        return images
    
    def noise_input(self, z, eps=None, timestep: Optional[int] = None, scheduler=None, t_min=0, t_max=999, alphas=None, sigmas=None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=t_min,
                high=min(t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = alphas.to(z.device)[timestep, None, None, None, None]
        sigma_t = sigmas.to(z.device)[timestep, None, None, None, None]
        print(f'alpha_t:{alpha_t.shape}, {z.shape}, {sigma_t.shape}, {eps.shape}')
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t
    
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        reference_control_writer = None,
        reference_control_reader = None,

        ref_latents: Optional[torch.FloatTensor] = None,
        ref_img_path: str = None,
        ref_prompt: Union[str, List[str]] = None,
        num_objects: int = None,
        fusion_blocks: str = "midup",

        use_reference_encoder: bool = True,

        p_min: float = 0,
        p_max: float = 1,

        p1: tuple[float] = [0, 0.3],
        p2: tuple[float] = [0.3, 0.6],
        p3: tuple[float] = [0.6, 1.0],
        
        t_min_dds: int = 0,
        t_max_dds: int = 50,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, 
            device, 
            num_videos_per_prompt, 
            do_classifier_free_guidance, 
            negative_prompt
        )

        if use_reference_encoder:
            reference_control_writer = ReferenceAttentionControl(self.reference_encoder, 
                                                                 do_classifier_free_guidance=True, 
                                                                 mode='write', 
                                                                 batch_size=1,
                                                                 fusion_blocks=fusion_blocks)
            reference_control_reader = ReferenceAttentionControl(self.unet, 
                                                                 do_classifier_free_guidance=True, 
                                                                 mode='read', 
                                                                 batch_size=1,
                                                                 fusion_blocks=fusion_blocks,
                                                                 num_objects=num_objects)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables

        num_channels_latents = self.unet.in_channels
        noise_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        latents = noise_latents
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # injection_prop = 0.9
        # t_inj = timesteps[int(injection_prop*len(timesteps))]
        # p_min, p_max = 0.1, 0.8
        # t_inj_max, t_inj_min = timesteps[int(p_min*(len(timesteps)-1))], timesteps[int(p_max*(len(timesteps)-1))]

        t1_start, t1_end = timesteps[int(p1[0]*(len(timesteps)-1))], timesteps[int(p1[1]*(len(timesteps)-1))]
        t2_start, t2_end = timesteps[int(p2[0]*(len(timesteps)-1))], timesteps[int(p2[1]*(len(timesteps)-1))]
        t3_start, t3_end = timesteps[int(p3[0]*(len(timesteps)-1))], timesteps[int(p3[1]*(len(timesteps)-1))]
        # print(f'injection between timesteps {t_inj_min} and {t_inj_max}')

        pred_z0 = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # use_ref_inj = t > t_inj
                
                with torch.no_grad():
                    use_p1_inj = (t <= t1_start and t > t1_end).item()
                    use_p2_inj = (t <= t2_start and t > t2_end).item()
                    use_p3_inj = (t <= t3_start and t >= t3_end).item()
                    print(f't:{t}, t1:{t1_start}, {t1_end}; t2:{t2_start}, {t2_end}; t3:{t3_start}, {t3_end}; {use_p1_inj}, {use_p2_inj}, {use_p3_inj}')
                    assert not (use_p1_inj+use_p2_inj+use_p3_inj)==3, f'only one or two models should be used in each timestep'
                    assert not (i == 0 and (use_p1_inj+use_p2_inj+use_p3_inj)>1), f'only one model should be used in start '

                    single = (use_p1_inj+use_p2_inj+use_p3_inj)==1
                    double = (use_p1_inj+use_p2_inj+use_p3_inj)==2
                    print(use_p1_inj+use_p2_inj+use_p3_inj)
                    print(f'double:{double}')
                    noise_dds = []
                    noise_pred_list = []

                    if pred_z0 is not None:
                        pred_z0_t, eps, timestep_dds, alpha_t, sigma_t = self.noise_input(pred_z0, 
                                                                                        scheduler=self.dds_scheduler, 
                                                                                        t_min=t_min_dds, 
                                                                                        t_max=t_max_dds,
                                                                                        alphas=torch.sqrt(self.scheduler.alphas_cumprod).to(latents.device, dtype=latents.dtype),
                                                                                        sigmas=torch.sqrt(1 - self.scheduler.alphas_cumprod).to(latents.device, dtype=latents.dtype))
                        alpha_exp, sigma_exp = 0, 0

                    # expand the latents if we are doing classifier free guidance
                    #print(f'do_classifier_free_guidance:{do_classifier_free_guidance}, latents:{latents.shape}')
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if use_p1_inj:
                        noise_pred_p1 = self.unet_phase2(
                                            latent_model_input, 
                                            t, 
                                            encoder_hidden_states=text_embeddings,
                                            ).sample.to(dtype=latents_dtype)
                        noise_pred_list.append(noise_pred_p1)
                        if double:
                            with torch.inference_mode():
                                noise_pred_p1_dds = self.unet_phase2(
                                                    torch.cat([pred_z0_t]*2), 
                                                    t, 
                                                    encoder_hidden_states=text_embeddings,
                                                    ).sample.to(dtype=latents_dtype)
                                noise_dds.append(noise_pred_p1_dds)

                    if use_p2_inj:
                        print(f'latent_model_input:{latent_model_input.shape}, t:{t.shape}, text_embedding:{text_embeddings.shape}')
                        noise_pred_p2 = self.unet_src(
                                            latent_model_input, 
                                            t, 
                                            encoder_hidden_states=text_embeddings,
                                            ).sample.to(dtype=latents_dtype)
                        noise_pred_list.append(noise_pred_p2)
                        if double:
                            with torch.inference_mode():
                                noise_pred_p2_dds = self.unet_src(
                                                    torch.cat([pred_z0_t]*2), 
                                                    t, 
                                                    encoder_hidden_states=text_embeddings,
                                                    ).sample.to(dtype=latents_dtype)
                                noise_dds.append(noise_pred_p2_dds)
                        
                    if use_p3_inj:
                        # referene encoder inference
                        if use_reference_encoder:
                            # Encode input prompt
                            ref_text_embeddings = self._encode_prompt(
                                ref_prompt, 
                                device, 
                                num_videos_per_prompt, 
                                do_classifier_free_guidance, 
                                negative_prompt=None
                            )

                            # Encode reference images
                            assert len(ref_img_path) == num_objects, f'Number of reference images ({len(ref_img_path)}) does not match number of objects ({num_objects})'
                            ref_latents = torch.cat([self.images2latents(np.array(Image.open(ref_img).convert('RGB').resize((width, height)))[None, :], latents_dtype).to(device) for ref_img in ref_img_path], dim=0)#(nobj*f,c,h,w)
                            #ref_text_embeddings = ref_text_embeddings.repeat_interleave(num_objects, 0)
                            self.reference_encoder(
                                            ref_latents.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1), 
                                            t, 
                                            encoder_hidden_states=ref_text_embeddings, 
                                            return_dict=False)

                            # predict the noise residual
                            reference_control_reader.update(reference_control_writer)

                            noise_pred_p3 = self.unet(
                                            latent_model_input, 
                                            t, 
                                            encoder_hidden_states=text_embeddings,
                                            ).sample.to(dtype=latents_dtype)
                            noise_pred_list.append(noise_pred_p3)
                            
                            if double:
                                with torch.inference_mode():
                                    noise_pred_p3_dds = self.unet(
                                                torch.cat([pred_z0_t]*2), 
                                                t, 
                                                encoder_hidden_states=text_embeddings,
                                                ).sample.to(dtype=latents_dtype)
                                    noise_dds.append(noise_pred_p3_dds)
                        
                            reference_control_reader.clear()
                        else:
                            raise ValueError(f'reference encoder should be used in p3')
                    

                # predict the denoising noise
                if single:
                    with torch.no_grad():
                        if use_p1_inj:
                            noise_pred = noise_pred_p1
                        if use_p2_inj:
                            noise_pred = noise_pred_p2
                        if use_p3_inj:
                            noise_pred = noise_pred_p3
                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        print(f'single: noise_pred:{noise_pred.shape}')

                elif double:
                    noise_pred_list_final = []
                    for noise_pred in noise_pred_list:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        noise_pred_list_final.append(noise_pred)

                    noise_dds_list = []
                    for noise_dds_item in noise_dds:
                        noise_dds_uncond, noise_dds_cond = noise_dds_item.chunk(2)
                        noise_dds_list.append(noise_dds_uncond + guidance_scale * (noise_dds_cond - noise_dds_uncond))
                    assert len(noise_dds_list) == 2, f'only two is needed'
                    grad = (alpha_t ** alpha_exp) * (sigma_t ** sigma_exp) * (noise_dds_list[0] - noise_dds_list[1])
                    loss_dds = pred_z0 * grad.clone()
                    loss_dds = loss_dds.sum() / ( pred_z0.shape[2] * pred_z0.shape[3] * pred_z0.shape[4] )
                    optimizer_dds.zero_grad()
                    loss_dds.backward()
                    grad_z0 = pred_z0.grad.data.detach()
                    grad_z0 = grad_z0.sum(1, keepdim=True) #(b, 1, f, h, w)
                    grad_z0 = (grad_z0 - grad_z0.min()) / (grad_z0.max() - grad_z0.min() + 1e-9) #(b, 1, f, h, w)
                    print(f'grad_z0:{grad_z0.shape}, {grad_z0.min()}, {grad_z0.max()}')
                    noise_pred = noise_pred_list_final[0] * (1. - grad_z0) + noise_pred_list_final[1] * grad_z0

                    # vis the grad mask
                    h, w = 512, 512
                    grad_target = grad_z0.squeeze(2) #(b, 1, h, w)
                    grad_target_list = []
                    grad_target_vis = F.interpolate(grad_target, (h,w), mode='nearest')
                    grad_target_vis = grad_target_vis.squeeze(1).permute(1,2,0).cpu().numpy()#(h,w,b)
                    for i in range(grad_target_vis.shape[-1]):
                        grad_target_item = plt.cm.viridis(grad_target_vis[:,:,i:i+1])#(h,w,1,4)
                        grad_target_item = (grad_target_item[:,:,0,:3])#(h,w,3)
                        # print(f'grad_target_item_{i}:{grad_target_item.min()}, {grad_target_item.max()}')
                        grad_target_item = torch.Tensor(grad_target_item)
                        grad_target_list.append(grad_target_item)
                    grad_target_tensor = torch.stack(grad_target_list, 0).permute(0, 3, 1, 2).contiguous()#(b,3,h,w)
                    print(f'grad_target_tensor:{grad_target_tensor.max()}')
                    os.makedirs(f'./tmp/mix_log', exist_ok=True)
                    torchvision.utils.save_image(torch.cat([grad_target_tensor], 0), f'./tmp/mix_log/{prompt}_{t:04}.png', nrow=2)

                else:
                    raise ValueError

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample.detach() # latents does't need grad
                print(f'latents:{latents.requires_grad}')
                alpha_t = torch.sqrt(self.scheduler.alphas_cumprod).to(latents.device, dtype=latents.dtype)[t, None, None, None]
                sigma_t = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(latents.device, dtype=latents.dtype)[t, None, None, None]
                pred_z0 = ((latents - noise_pred * sigma_t) / alpha_t).detach()
                pred_z0.requires_grad = True
                optimizer_dds = SGD(params=[pred_z0], lr=1e-1)

                if use_reference_encoder and use_p3_inj:
                    reference_control_writer.clear()
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # # Post-processing
        with torch.no_grad():
            video = self.decode_latents(latents)
            image = rearrange(video, 'b c f h w -> (b f) c h w')

            # Convert to tensor
            if output_type == "tensor":
                video = torch.from_numpy(video)
                image = torch.from_numpy(image)
                

            if not return_dict:
                return video, image

            # ref_image_tensor = self.images2tensor(np.array(Image.open(ref_img).convert('RGB').resize((width, height)))[None, :], latents_dtype)
            ref_image_tensor = torch.cat([self.images2tensor(np.array(Image.open(ref_img).convert('RGB').resize((width, height)))[None, :], latents_dtype).to(device) for ref_img in ref_img_path], dim=0)#(nobj*f,c,h,w)
            return LAMPPipelineOutput(videos=video, image=image, ref_image_tensor=ref_image_tensor)
