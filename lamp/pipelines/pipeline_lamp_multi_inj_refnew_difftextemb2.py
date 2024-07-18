# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from tqdm import tqdm

from math import sqrt
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from lamp.util import save_videos_grid, ddim_inversion

from lamp.models.controlnet import ControlNetModel

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
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

from lamp.models.mutual_self_attention_refnew import ReferenceAttentionControl
from lamp.models.reference_encoder import AppearanceEncoderModel
from lamp.models.cross_attention_control import CrossAttentionControl

from lamp.tools import *

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
    image_ref: Union[torch.Tensor, np.ndarray]


class LAMPPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        reference_encoder: Union[AppearanceEncoderModel, None],
        controlnet: Optional[ControlNetModel] = None,
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

        if controlnet is not None:
            if reference_encoder is not None:
                self.register_modules(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    reference_encoder=reference_encoder,
                    controlnet=controlnet,
                    scheduler=scheduler,
                )
            else:
                self.register_modules(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    controlnet=controlnet,
                    scheduler=scheduler,
                )
                self.reference_encoder = None
        else:
            if reference_encoder is not None:
                self.register_modules(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    reference_encoder=reference_encoder,
                    scheduler=scheduler,
                )
            else:
                self.register_modules(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=scheduler,
                )
                self.reference_encoder = None
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

    def obtain_text(self, text, object_category=None):
        if object_category is None:
            placeholder_string = self.placeholder_token
        else:
            placeholder_string = object_category
        
        if isinstance(text, str):
            text = [text]
        placeholder_index = 0

        index = []
        for text_item in text:
            words = text_item.strip().split(' ')
            print(words, placeholder_string)
            for idx, word in enumerate(words):
                if word == placeholder_string:
                    placeholder_index = idx+1 #has start token
        
            index.append(torch.tensor(placeholder_index))
        index = torch.stack(index)

        text_inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )#(n, 77)
        return text_inputs, index

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, inj_embeddings=None, inj_index=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if self.separate_v:
            text_inputs, cond_index = self.obtain_text(prompt, self.placeholder_token)
        else:
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
            {'input_ids': text_input_ids.to(device),
             'inj_embedding': inj_embeddings.to(device) if inj_embeddings is not None else inj_embeddings,
             'inj_index': inj_index},
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        if self.separate_v:
            cond_index = cond_index.unsqueeze(1).repeat(1, num_videos_per_prompt).view(-1, 1).squeeze(1)

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
                {'input_ids': uncond_input.input_ids.to(device)},
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

            if self.separate_v:
                uncond_index = torch.tensor([0]*uncond_embeddings.shape[0])
                index = torch.cat([uncond_index, cond_index])
                return text_embeddings, index

        return text_embeddings, None

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
    def freq2embeddings(self, freq_embeddings, bsz, dtype):
        freq_embeddings_input = torch.repeat_interleave(freq_embeddings.embedding.weight.unsqueeze(1), bsz, 0)
        inj_embeddings = combine_embeddings(freq_embeddings_input, flag_to_combine=torch.ones(1)*2)
        return inj_embeddings
    
    @torch.no_grad()
    def text2injindex(self, text, placeholder_string):
        placeholder_index = 0
        words = text.strip().split(' ')
        # print(words, placeholder_string)
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx+1 #has start token
        
        index = torch.tensor(placeholder_index)
        return torch.stack([index])

    @torch.no_grad()
    def images2tensor(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "f h w c -> f c h w").to(device)

        return images

    @torch.no_grad()
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

        controlnet_cond: Optional[torch.FloatTensor] = None,

        use_reference_encoder: bool = True,

        p_min: int = 0,
        p_max: int = 1,

        separate_v: bool = False,
        placeholder_token: str = '*',

        high_freq_percentage: float = 5,
        low_freq_percentage: float = 5,

        freq_embeddings: torch.tensor = None,

        style_prompt: Union[str, List[str]] = None,

        **kwargs,
    ):
        self.separate_v = separate_v
        self.placeholder_token = placeholder_token

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
        inj_embeddings = self.freq2embeddings(
            freq_embeddings,
            batch_size,
            torch.float32,
        )
        inj_index = self.text2injindex(prompt, self.placeholder_token)
        text_embeddings, text_index = self._encode_prompt(
            prompt, 
            device, 
            num_videos_per_prompt, 
            do_classifier_free_guidance, 
            negative_prompt,
            inj_embeddings=inj_embeddings,
            inj_index=inj_index,
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

        latents = noise_latents #(b, c, t, h, w)
        #print(f'noise_latents:{latents.sum([1,2,3,4])}, {latents.shape}')
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if self.separate_v:
            cross_attention_controller = CrossAttentionControl(self.unet, fusion_blocks="full")

        # injection_prop = 0.9
        # t_inj = timesteps[int(injection_prop*len(timesteps))]
        # p_min, p_max = 0.1, 0.8
        t_inj_max, t_inj_min = timesteps[int(p_min*(len(timesteps)-1))], timesteps[int(p_max*(len(timesteps)-1))]
        #print(f'injection between timesteps {t_inj_min} and {t_inj_max}')
        ref_encoded = False
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # use_ref_inj = t > t_inj
                use_ref_inj = (t >= t_inj_min and t <= t_inj_max)
                # controlnet inference
                down_block_res_samples, mid_block_res_sample = None, None

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                #print(f'latent_model_input:{latent_model_input.sum([1,2,3,4])},{latent_model_input.shape}')
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                #print(f'scale latent_model_input:{latent_model_input.sum([1,2,3,4])},{latent_model_input.shape}')
                # referene encoder inference
                if use_reference_encoder and use_ref_inj: 
                    if not ref_encoded:
                        # Encode input prompt
                        ref_text_embeddings, _ = self._encode_prompt(
                            ref_prompt, 
                            device, 
                            num_videos_per_prompt, 
                            do_classifier_free_guidance, 
                            negative_prompt=None
                        )

                        # Encode reference images
                        assert len(ref_img_path) == num_objects, f'Number of reference images ({len(ref_img_path)}) does not match number of objects ({num_objects})'
                        ref_latents = torch.cat([self.images2latents(np.array(Image.open(ref_img).convert('RGB').resize((width, height)))[None, :], latents_dtype).to(device) for ref_img in ref_img_path], dim=0)#(nobj*f,c,h,w)
                        ref_encoded = True
                        
                        style_latents = latent_model_input.squeeze(2) #(b, c, h, w)
                        # print(f'style_latents1:{style_latents.sum([1,2,3])}, {style_latents.shape}')
                    #ref_text_embeddings = ref_text_embeddings.repeat_interleave(num_objects, 0)
                    # print(f'style_latents2:{style_latents.shape}')
                    
                    #ref_latents = self.scheduler.add_noise(ref_latents, torch.randn_like(ref_latents), t)

                    reference_latent_input = torch.cat([ref_latents.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1), style_latents], dim=0)
                    # print(f'reference_latent_input:{reference_latent_input.shape}')
                    
                    noise_pred_ref = self.reference_encoder(
                                      reference_latent_input, 
                                      t, 
                                      encoder_hidden_states=torch.cat([ref_text_embeddings]*2), #torch.cat([ref_text_embeddings]*2), 
                                      return_dict=True).sample
                    #print(f'timestep:{t}, reference_latent_input:{reference_latent_input.sum([1,2,3])}, encoder_hidden_states:{torch.cat([ref_text_embeddings, style_text_embeddings], 0).sum([1,2])}, noise_pred_ref:{noise_pred_ref.sum([1,2,3])}')
                    # print(f'noise_pred_ref:{noise_pred_ref.shape}')
                    _, noise_pred_style = noise_pred_ref.chunk(2)
                    if do_classifier_free_guidance:
                        noise_pred_style_uncond, noise_pred_style_text = noise_pred_style.chunk(2)
                        noise_pred_style = noise_pred_style_uncond + guidance_scale * (noise_pred_style_text - noise_pred_style_uncond)
                    # compute the previous noisy sample x_t -> x_t-1
                    # print(f'noise_pred_style:{noise_pred_style.shape}, style_latents:{style_latents.shape}')
                    b = style_latents.shape[0]
                    style_latents_denoise = self.scheduler.step(noise_pred_style, t, style_latents[:b//2] if do_classifier_free_guidance else style_latents, **extra_step_kwargs).prev_sample
                    style_latents = torch.cat([style_latents_denoise]*2, 0)  if do_classifier_free_guidance else style_latents_denoise

                if self.separate_v:
                    cross_attention_controller.update(text_index)
                    print(f'update cross attention in inference')

                # predict the noise residual
                if use_reference_encoder and use_ref_inj:
                    reference_control_reader.update(reference_control_writer)
                noise_pred = self.unet(
                                latent_model_input, 
                                t, 
                                encoder_hidden_states=text_embeddings,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                )
                noise_pred = noise_pred.sample.to(dtype=latents_dtype)
                if use_reference_encoder and use_ref_inj:
                    reference_control_reader.clear()

                if self.separate_v:
                    cross_attention_controller.clear()
                    print(f'clear cross attention in inference')

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if use_reference_encoder and use_ref_inj:
                    reference_control_writer.clear()
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # # Post-processing
        video = self.decode_latents(latents)
        image = rearrange(video, 'b c f h w -> (b f) c h w')

        video_style = self.decode_latents(style_latents_denoise.unsqueeze(2))
        image_ref = rearrange(video_style, 'b c f h w -> (b f) c h w')

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)
            image = torch.from_numpy(image)
            image_ref = torch.from_numpy(image_ref)
            

        if not return_dict:
            return video, image

        # ref_image_tensor = self.images2tensor(np.array(Image.open(ref_img).convert('RGB').resize((width, height)))[None, :], latents_dtype)
        ref_image_tensor = torch.cat([self.images2tensor(np.array(Image.open(ref_img).convert('RGB').resize((width, height)))[None, :], latents_dtype).to(device) for ref_img in ref_img_path], dim=0)#(nobj*f,c,h,w)
        return LAMPPipelineOutput(videos=video, image=image, ref_image_tensor=ref_image_tensor, image_ref=image_ref)
