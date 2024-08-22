import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from einops import rearrange
import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt

import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import torchvision.utils as T
import torchvision.utils as TU
from torchvision.utils import save_image
import torchvision

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from omegaconf import OmegaConf

import diffusers
from diffusers.utils import export_to_gif
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

import transformers
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING
from packaging import version
if version.parse(transformers.__version__) > version.parse('4.32.0'):
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
else:
    from transformers.models.clip.modeling_clip import _expand_mask

from eqdiff.pipelines.pipeline_eqdiff_multi_inj_refnew_difftextemb2 import EQDIFFPipeline
from eqdiff.models import ptp_utils
from eqdiff.models.ptp_utils import AttentionStore
from eqdiff.models.unet import UNet3DConditionModel
from eqdiff.models.reference_encoder import AppearanceEncoderModel
from eqdiff.models.mutual_self_attention_refnew import ReferenceAttentionControl
from eqdiff.models.attention_processor_custom import CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor
from eqdiff.models.align_hook import AlignLossHook, MasksHook
from eqdiff.models.self_attention_loss import SALoss
from eqdiff.models.cross_attention_control import CrossAttentionControl
from eqdiff.tools import *
from eqdiff.util import save_videos_grid, ddim_inversion, load_weights_into_unet

from dataloader.reference_diffusion_dataset import collate_fn, PromptDataset, ReferenceDiffusionDataset


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0")

logger = get_logger(__name__)

res_max = 16

def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = vae.decode(latents).sample
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    video = video.cpu().float().numpy()
    return video

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def unfreeze_params(params):
    for param in params:
        param.requires_grad = True

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def save_new_embed(text_encoder, modifier_token_id, accelerator, args, output_dir, safe_serialization=True):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        filename = f"{output_dir}/{y}.bin"

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, filename, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, filename)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Custom Diffusion training script.")
    parser.add_argument("--config", type=str, default="./configs/custom_diffusion.yaml")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--reference_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained reference model.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path_for_img_gen",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_clip_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model of clip model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument(
        "--enable_phase3_training",
        default=False,
        action="store_true",
        help="enable phase3 training"
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--align_loss_weight", type=float, default=1.0, help="The weight of alignment loss for residual self-attention module.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eqdiff-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--phase1_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the first phase.",
    )
    parser.add_argument(
        "--phase2_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the second phase.",
    )
    parser.add_argument(
        "--phase3_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the second phase.",
    )
    parser.add_argument(
        "--phase4_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the second phase.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--phase3_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--phase4_learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=5e-4,
        help="The LR for the Textual Inversion steps.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default="crossattn_kv",
        choices=["crossattn_kv", "crossattn"],
        help="crossattn to enable fine-tuning of all params in the cross attention",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--text_inj",
        action="store_true",
        help="Apply customized text embedding injection.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--training_log_steps",
        type=int,
        default=50,
        help="Number of iterations to log in training",
    )
    parser.add_argument(
        "--only_train_k",
        action="store_true",
        help="only train k in cross attention.",
    )
    parser.add_argument(
        "--only_train_v",
        action="store_true",
        help="only train v in cross attention.",
    )
    parser.add_argument(
        "--separate_v",
        action="store_true",
        help="only train v for rare token in cross attention.",
    )
    parser.add_argument(
        "--mask_noise",
        action="store_true",
        help="only add noise in mask region.",
    )
    parser.add_argument("--use_reference_encoder", action="store_true", help="if use reference encoder")
    parser.add_argument("--mask_noise_only_loss", action="store_true", help="only apply mask noise in loss calculation")
    parser.add_argument("--mask_noise_prob", type=float, default=1., help='mask noise prob')
    parser.add_argument("--dilate_iters", type=int, default=3, help="dilate mask iterations")
    parser.add_argument("--lambda_attention", type=float, default=1e-2)
    parser.add_argument("--lambda_sattention", type=float, default=1e-2)
    parser.add_argument("--high_freq_percentage", type=int, default=5, help="high_freq_percentage")
    parser.add_argument("--low_freq_percentage", type=int, default=5, help="low_freq_percentage")
    parser.add_argument("--gray_rate", type=float, default=0.5, help="high_freq_percentage")
    parser.add_argument("--grayprob", type=float, default=0, help="prob to use gray")
    parser.add_argument("--use_gray", action="store_true", help="If specified use gray token to disentangle the high frequency part")
    parser.add_argument("--time_gap", type=int, default=2, help="gap between reference and main ddpm")
    parser.add_argument("--LF", type=float, default=0.3, help="prob to use lf")
    parser.add_argument("--HF", type=float, default=0.3, help="prob to use hf")
    parser.add_argument("--AllF", type=float, default=0.6, help="prob to use allf")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.enable_phase3_training:
        args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps + args.phase3_train_steps + args.phase4_train_steps
    else:
        args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    r_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = r_input_ids.size()
    r_input_ids = r_input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx in enumerate(inj_index):
            if not idx == 0:# prior preservation use init index=0, should skip this 
                lll = new_inputs_embeds[bsz, idx+emb_length:].shape[0]
                new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+lll]
                new_inputs_embeds[bsz, idx:idx+emb_length] = inj_embedding[bsz]

    hidden_states = self.embeddings(input_ids=r_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = AttentionMaskConverter()._expand_mask(attention_mask, hidden_states.dtype)
        # attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def main(args):
    token_embed_prev = None
    print(args.validation_prompt)
    print(args.num_validation_images)
    args_addition = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # args.output_dir = os.path.join(args.output_dir, current_time)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("eqdiff", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept["class_data_dir"])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if args.real_prior:
                assert (
                    class_images_dir / "images"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                    len(list((class_images_dir / "images").iterdir())) == args.num_class_images
                ), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                    class_images_dir / "caption.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                assert (
                    class_images_dir / "images.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
                concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
                args.concepts_list[i] = concept
                accelerator.wait_for_everyone()
            else:
                cur_class_images = len(list(class_images_dir.iterdir()))

                if cur_class_images < args.num_class_images:
                    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                    if args.prior_generation_precision == "fp32":
                        torch_dtype = torch.float32
                    elif args.prior_generation_precision == "fp16":
                        torch_dtype = torch.float16
                    elif args.prior_generation_precision == "bf16":
                        torch_dtype = torch.bfloat16
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path_for_img_gen,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = args.num_class_images - cur_class_images
                    logger.info(f"Number of class images to sample: {num_new_images}.")

                    print(args.class_prompt)
                    sample_dataset = PromptDataset(args.class_prompt, num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                    sample_dataloader = accelerator.prepare(sample_dataloader)
                    pipeline.to(accelerator.device)

                    for example in tqdm(
                        sample_dataloader,
                        desc="Generating class images",
                        disable=not accelerator.is_local_main_process,
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = (
                                class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            )
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # replace the forward method of the text encoder to inject the word embedding
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )
    unet_additional_kwargs = args_addition['unet_additional_kwargs']
    ref_encoder_addition_kwargs = args_addition['ref_encoder_addition_kwargs']
    num_objects_given, validation_num_objects = args_addition['num_objects_given'], args_addition['validation_num_objects']
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="unet", unet_additional_kwargs=unet_additional_kwargs)

    ## loading and setting reference encoder 
    print(f'loading and setting reference encoder')
    use_reference_encoder = True if args.use_reference_encoder else False
    if use_reference_encoder:
        reference_encoder = AppearanceEncoderModel.from_pretrained(ref_encoder_addition_kwargs['pretrained_model_path'], subfolder="unet")

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id = tokenizer.convert_tokens_to_ids(args.modifier_token)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        
        # add two token embeddings to represent hf and lf
        _, embedding_dim = token_embeds.shape
        freq_embeddings = LearnableEmbeddings(2, embedding_dim)#(2,768) hf concat lf

        
        for token_id in modifier_token_id:
            with torch.no_grad():
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()

            freq_embeddings.embedding.weight.data.copy_(torch.cat([token_embeds[initializer_token_id]]*2))
            #for _ in range(2):
            #    freq_embeddings.append(token_embeds[initializer_token_id].clone())
        #freq_embeddings = torch.cat(freq_embeddings)#(2,768) hf concat lf
        #freq_embeddings.requires_grad_(True)
        print(f'start freq_embeddings:{freq_embeddings.embedding.weight.shape}, {freq_embeddings.embedding.weight.sum(1)}, {freq_embeddings.embedding.weight.requires_grad}')
        # unfreeze_params(freq_embeddings)

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
    ########################################################
    ########################################################

    vae.requires_grad_(False)
    if args.modifier_token is None:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16" and args.modifier_token is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    # unet.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)
    vae.to(accelerator.device, dtype=weight_dtype)

    attention_class = CustomDiffusionAttnProcessor
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            attention_class = CustomDiffusionXFormersAttnProcessor
            print(f'use CustomDiffusionXFormersAttnProcessor')

            if use_reference_encoder:
                reference_encoder.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if use_reference_encoder:
            reference_encoder.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.initial_learning_rate = (
            args.initial_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.phase3_learning_rate = (
            args.phase3_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.phase4_learning_rate = (
            args.phase4_learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0
            args.phase3_learning_rate = args.phase3_learning_rate * 2.0
            args.phase4_learning_rate = args.phase4_learning_rate * 2.0
            args.initial_learning_rate = args.initial_learning_rate * 2.0

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # params = [{'params':itertools.chain(text_encoder.get_input_embeddings().parameters()), 'lr': args.initial_learning_rate}]
    
    optimizer = optimizer_class(
        #itertools.chain(text_encoder.get_input_embeddings().parameters()),
        itertools.chain(freq_embeddings.parameters()),
        lr=args.initial_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = ReferenceDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        mask_size=vae.encode(
            torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
        ).latent_dist.sample().size()[-1],
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip,
        aug=not args.noaug,
        num_ref=1, #num_objects_given,
        placeholder_token=args.modifier_token[0],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if args.modifier_token is not None:
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    num_objects_given_prev = None

    # self attention loss 
    saloss = SALoss()
    img_logs_path = os.path.join(args.output_dir, 'attn_loss')
    os.makedirs(img_logs_path, exist_ok=True)

    # mask hook for mask fusion
    mask_hook = MasksHook()
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        models_accumulate = [unet]
        if args.modifier_token is not None:
            text_encoder.train()
            models_accumulate.append(text_encoder)            
        for step, batch in enumerate(train_dataloader):
            cur_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
            # print(f"text embed changed? {orig_embeds_params[:49408] == cur_embeds_params[:49408]}")
            use_gray = False if not args.use_gray else generate_random_bool(grayprob=args.grayprob)
            # two stage training

            # extra settings
            mask_noise = generate_random_bool(args.mask_noise_prob)
            
            # self-attention res training settings
            att_count = 0
            train_res = True
            use_align_loss = False
            use_integra_ref_mask = True
            loss_weights = {"down_self": 0.2,"mid_self": 0.4, "up_self": 1.0 }
            
            phase2_training = global_step >= args.phase1_train_steps
            phase3_training = (global_step >= (args.phase1_train_steps + args.phase2_train_steps)) and args.enable_phase3_training
            
            # separate v between original and added values
            separate_v = args.separate_v
            if separate_v:
                cross_attention_controller = CrossAttentionControl(unet, fusion_blocks="full")

            if global_step == args.phase1_train_steps:
                logger.info("Start Phase 2 training")
                # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
                
                train_kv = True
                train_q_out = False if args.freeze_model == "crossattn_kv" else True
                custom_diffusion_attn_procs = {}

                st = unet.state_dict()
                for name, _ in unet.attn_processors.items():
                    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                    if name.startswith("mid_block"):
                        hidden_size = unet.config.block_out_channels[-1]
                        place_in_unet = "mid"
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                        place_in_unet = "up"
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = unet.config.block_out_channels[block_id]
                        place_in_unet = "down"
                    else:
                        raise ValueError(f"unexpected layer name: {name}")
                        
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
                        "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
                    }
                    if train_q_out:
                        weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
                        weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
                        weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
                    if cross_attention_dim is not None:
                        # cross attention module
                        custom_diffusion_attn_procs[name] = attention_class(
                            train_kv=train_kv,
                            only_train_k=args.only_train_k,
                            only_train_v=args.only_train_v,
                            train_q_out=train_q_out,
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                        ).to(unet.device)
                        custom_diffusion_attn_procs[name].load_state_dict(weights)
                    else:
                        # self attention module
                        custom_diffusion_attn_procs[name] = attention_class(
                                train_kv=False,
                                train_q_out=False,
                                hidden_size=hidden_size,
                                cross_attention_dim=cross_attention_dim,
                            )
                del st

                unet.set_attn_processor(custom_diffusion_attn_procs)
                custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
                accelerator.register_for_checkpointing(custom_diffusion_layers)

                # print(f'phase2:', [k for k in unet.state_dict().keys() if 'custom' in k ])

                del optimizer 
                if args.modifier_token is not None:
                    #parameters_to_optimize = [text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters()]
                    parameters_to_optimize = [freq_embeddings.parameters(), custom_diffusion_layers.parameters()]
                else:
                    parameters_to_optimize = [custom_diffusion_layers.parameters()]
                
                optimizer = optimizer_class(
                    itertools.chain(*parameters_to_optimize),
                    lr=args.learning_rate,
                    # params = params,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )

                del lr_scheduler
                lr_scheduler = get_scheduler(
                    args.lr_scheduler,
                    optimizer=optimizer,
                    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                    num_training_steps=args.max_train_steps * accelerator.num_processes,
                )

                # Prepare everything with our `accelerator`.
                custom_diffusion_layers, optimizer, lr_scheduler = accelerator.prepare(
                    custom_diffusion_layers, optimizer, lr_scheduler
                )

            if (global_step == args.phase1_train_steps + args.phase2_train_steps) and phase3_training:           
                logger.info("Start Phase 3 training")
                # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
                
                train_kv = True
                train_q_out = False if args.freeze_model == "crossattn_kv" else True
                custom_diffusion_attn_procs = {}

                controller = AttentionStore()
                controller_res = AttentionStore()

                st = unet.state_dict()
                for name, _ in unet.attn_processors.items():
                    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                    if name.startswith("mid_block"):
                        hidden_size = unet.config.block_out_channels[-1]
                        place_in_unet = "mid"
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                        place_in_unet = "up"
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = unet.config.block_out_channels[block_id]
                        place_in_unet = "down"
                    else:
                        raise ValueError(f"unexpected layer name: {name}")
                        
                    layer_name = name.split(".processor")[0]
                    
                    # print(f'phase3:', [k for k in st.keys() if 'custom' in k ])
                    # print(name, layer_name, cross_attention_dim is None)
                    if cross_attention_dim is not None:
                        # cross attention module
                        weights = {
                            "to_k_custom_diffusion.weight": st[name + ".to_k_custom_diffusion.weight"],
                            "to_v_custom_diffusion.weight": st[name + ".to_v_custom_diffusion.weight"],
                        }
                        if train_q_out:
                            weights["to_q_custom_diffusion.weight"] = st[name + ".to_q_custom_diffusion.weight"]
                            weights["to_out_custom_diffusion.0.weight"] = st[name + ".to_out_custom_diffusion.0.weight"]
                            weights["to_out_custom_diffusion.0.bias"] = st[name + ".to_out_custom_diffusion.0.bias"]                        
                        custom_diffusion_attn_procs[name] = attention_class(
                            train_kv=train_kv,
                            only_train_k=args.only_train_k,
                            only_train_v=args.only_train_v,
                            train_q_out=train_q_out,
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            place_in_unet=place_in_unet,
                            controller=controller,
                            controller_res=None,
                        ).to(unet.device)
                        custom_diffusion_attn_procs[name].load_state_dict(weights)
                    else:
                        # self attention module
                        custom_diffusion_attn_procs[name] = attention_class(
                            train_kv=False,
                            train_q_out=False,
                            train_res=train_res,
                            use_integra_ref_mask=use_integra_ref_mask,
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            mask_hook = mask_hook,
                            place_in_unet=place_in_unet,
                            controller=controller,
                            controller_res=controller_res,
                        )
                        if train_res:
                            weights_res = {
                                "to_k_res.weight": st[layer_name + ".to_k.weight"],
                                "to_v_res.weight": st[layer_name + ".to_v.weight"],
                                "to_out_res.0.weight": custom_diffusion_attn_procs[name].to_out_res[0].weight,#should be zero
                                "to_out_res.0.bias": custom_diffusion_attn_procs[name].to_out_res[0].bias,
                            }
                            assert weights_res["to_out_res.0.weight"].sum() == 0
                            assert weights_res["to_out_res.0.bias"].sum() == 0
                            custom_diffusion_attn_procs[name].load_state_dict(weights_res)
                    att_count += 1
                controller.num_att_layers = 12 #4+2+6
                print(f'controller.num_att_layers:{controller.num_att_layers}, att_count:{att_count}')
                if ref_encoder_addition_kwargs['fusion_blocks'] == 'midup':
                    controller_res.num_att_layers = (1+3)# since only self-attention and mid_up (reference encoder setting) and only < res_max*res_max
                else:
                    raise ValueError(f'fusion method is not defined.')
                del st

                unet.set_attn_processor(custom_diffusion_attn_procs)
                custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
                accelerator.register_for_checkpointing(custom_diffusion_layers)


                # freeze the parameters of text_embeddings and text_encoder
                for name, processor in unet.attn_processors.items():
                    # print(name, processor)
                    if name.endswith("attn2.processor"):# cross attention
                        for name_p, param in processor.named_parameters():
                            param.requires_grad = False 
                    
                del optimizer 
                if args.modifier_token is not None:
                    #parameters_to_optimize = [text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters()]
                    parameters_to_optimize = [freq_embeddings.parameters(), custom_diffusion_layers.parameters()]
                else:
                    parameters_to_optimize = [custom_diffusion_layers.parameters()]

                optimizer = optimizer_class(
                    itertools.chain(*parameters_to_optimize),
                    lr=args.phase3_learning_rate,
                    # params = params,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )

                del lr_scheduler
                lr_scheduler = get_scheduler(
                    args.lr_scheduler,
                    optimizer=optimizer,
                    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                    num_training_steps=args.max_train_steps * accelerator.num_processes,
                )

                # Prepare everything with our `accelerator`.
                custom_diffusion_layers, optimizer, lr_scheduler = accelerator.prepare(
                    custom_diffusion_layers, optimizer, lr_scheduler
                )

            if (global_step == args.phase1_train_steps + args.phase2_train_steps + args.phase3_train_steps):
                # Save the custom diffusion layers of Text-Embedding and 
                accelerator.wait_for_everyone()

                logger.info("Start Phase 4 training")
                # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
                
                # freeze the parameters of text_embeddings and text_encoder
                for name, processor in unet.attn_processors.items():
                    # print(name, processor)
                    if name.endswith("attn2.processor"):# cross attention
                        for name_p, param in processor.named_parameters():
                            if "custom_diffusion" in name_p:
                                param.requires_grad = True 
                    
                del optimizer 
                if args.modifier_token is not None:
                    #parameters_to_optimize = [text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters()]
                    parameters_to_optimize = [freq_embeddings.parameters(), custom_diffusion_layers.parameters()]
                else:
                    parameters_to_optimize = [custom_diffusion_layers.parameters()]

                # params = [{"params": text_encoder.get_input_embeddings().parameters(), "lr": args.initial_learning_rate},
                #           {"params": custom_diffusion_layers.parameters(), "lr": args.phase4_learning_rate}] 
                optimizer = optimizer_class(
                    itertools.chain(*parameters_to_optimize),
                    lr=args.phase4_learning_rate,
                    # params = params,
                    betas=(args.adam_beta1, args.adam_beta2),
                    weight_decay=args.adam_weight_decay,
                    eps=args.adam_epsilon,
                )

                del lr_scheduler
                lr_scheduler = get_scheduler(
                    args.lr_scheduler,
                    optimizer=optimizer,
                    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                    num_training_steps=args.max_train_steps * accelerator.num_processes,
                )

                # Prepare everything with our `accelerator`.
                if use_reference_encoder:
                    reference_encoder = AppearanceEncoderModel.from_pretrained(args.reference_encoder_path, subfolder="unet")
                    reference_encoder.requires_grad_(False)
                    reference_encoder.to(accelerator.device)
                    # reference_encoder.train()
                    # models_accumulate.append(reference_encoder)
                    custom_diffusion_layers, reference_encoder, optimizer, lr_scheduler = accelerator.prepare(
                        custom_diffusion_layers, reference_encoder, optimizer, lr_scheduler
                    )
                    # print trainable modules in unet
                    trainable_params_name = []
                    for name_m, module in reference_encoder.named_modules():
                        for name_p, param in module.named_parameters():
                            if param.requires_grad:
                                trainable_params_name.append(f"{name_m}.{name_p}")
                    print(f'trainable_params_name in reference_encoder:{trainable_params_name}')
                else:
                    custom_diffusion_layers, optimizer, lr_scheduler = accelerator.prepare(
                        custom_diffusion_layers, optimizer, lr_scheduler
                    )

            if phase3_training and use_reference_encoder and (num_objects_given_prev != num_objects_given):
                reference_control_writer = ReferenceAttentionControl(reference_encoder, 
                                                                    do_classifier_free_guidance=False, 
                                                                    mode='write', 
                                                                    fusion_blocks=ref_encoder_addition_kwargs['fusion_blocks'])
                reference_control_reader = ReferenceAttentionControl(unet, 
                                                                    do_classifier_free_guidance=False, 
                                                                    mode='read', 
                                                                    fusion_blocks=ref_encoder_addition_kwargs['fusion_blocks'], 
                                                                    num_objects=num_objects_given)

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
            with accelerator.accumulate(models_accumulate):
                # Convert images to latent space
                if use_gray:
                    flag_to_combine = random.choices([0, 1, 2], weights=[args.LF, args.HF, args.AllF], k=batch["pixel_values"].shape[0])
                    pixel_values_freq, pixel_values = split_fft_values(
                                            batch["pixel_values"].to(dtype=weight_dtype), 
                                            flag_to_combine,
                                            args.high_freq_percentage,
                                            args.low_freq_percentage)
                else:
                    pixel_values = batch["pixel_values"]
                latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps_ref = torch.clamp(timesteps - args.time_gap, min=0)
                timesteps = timesteps.long()
                timesteps_ref = timesteps_ref.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if mask_noise:
                    dilate_iters = args.dilate_iters
                    dilated_mask = batch["instance_mask"]
                    for i, _ in enumerate(range(dilate_iters)):
                        dilated_mask = dilate(dilated_mask, ksize=5)
                    if not args.mask_noise_only_loss:# only apply mask noise in the loss calculation
                        noisy_latents = dilated_mask*noisy_latents + (1-dilated_mask)*latents

                # Get the text embedding for conditioning
                if not args.text_inj:
                    print(f'Not using text inj')
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]#(2,77,768)
                else:
                    if use_gray:
                        freq_embeddings_input = torch.repeat_interleave(freq_embeddings.embedding.weight.unsqueeze(1), bsz, 0)#(b*2, 1, c)
                        inj_embedding = combine_embeddings(freq_embeddings_input, flag_to_combine)#(b, 1, c)
                    else:
                        inj_embedding = None

                    encoder_hidden_states = text_encoder(
                        {
                        'input_ids': batch["input_ids"],
                        'inj_index': batch["index"].detach(),
                        'inj_embedding': inj_embedding,
                        }
                    )[0]

                # Predict the noise residual and compute loss
                if use_reference_encoder and phase3_training:
                    #assert num_objects_given == batch["pixel_values_ref"].shape[1],f'num_objects_given:{num_objects_given} vs {batch["pixel_values_ref"].shape[1]}'
                    pixel_values_ref = rearrange(batch["pixel_values_ref"].to(dtype=weight_dtype), "b n c h w -> (b n) c h w")
                    ref_encoder_text_ids = rearrange(batch["input_ids_ref"], "b n l -> (b n) l")
                    ref_encoder_text_embeddings = text_encoder({'input_ids':ref_encoder_text_ids})[0]
                    latents_ref = vae.encode(pixel_values_ref).latent_dist.sample()
                    latents_ref = latents_ref * vae.config.scaling_factor
                    #assert num_objects_given == latents.shape[2]# 
                    #print(f'latents:{latents.shape}, latents_ref:{latents_ref.shape}, timesteps:{timesteps.shape}, ref_encoder_text_embeddings:{ref_encoder_text_embeddings.shape}')
                    if True:
                        timesteps_ref = timesteps_ref.repeat_interleave(num_objects_given, 0)
                        noisy_latents_ref = noise_scheduler.add_noise(latents_ref, noise, timesteps_ref)
                        reference_encoder(torch.cat([noisy_latents_ref]*2, 0), torch.cat([timesteps_ref]*2, 0), torch.cat([ref_encoder_text_embeddings]*2, 0), return_dict=False)
                    else:
                        timesteps_ref = timesteps.repeat_interleave(num_objects_given, 0)
                        reference_encoder(torch.cat([latents_ref]*2, 0), torch.cat([timesteps_ref]*2, 0), torch.cat([ref_encoder_text_embeddings]*2, 0), return_dict=False)
                    reference_control_reader.update(reference_control_writer)

                if separate_v:
                    cross_attention_controller.clear()
                    cross_attention_controller.update(batch["index"])
                    print(f'update cross attention')

                # Predict the noise residual
                noisy_latents = rearrange(noisy_latents, '(b f) c h w -> b c f h w', f=1).contiguous()
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                model_pred = rearrange(model_pred, 'b c f h w -> (b f) c h w').contiguous()

                if separate_v:
                    cross_attention_controller.clear()

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
                    target_prior, target = torch.chunk(target, 2, dim=0)
                    if mask_noise:# use instance mask
                        mask = torch.chunk(dilated_mask,  2, dim=0)[1]
                    else:
                        mask = torch.chunk(batch["mask"], 2, dim=0)[1]
                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss and align loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    mask = batch["mask"]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                
                logs = {}

                # logging the training result
                if global_step % args.training_log_steps == 0:
                    if accelerator.is_main_process:
                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                #rearrange(decode_latents(vae, model_pred), )
                                #np_images = np.concatenate([np.asarray(img) for img in model_pred])#(b,3,h,w)
                                input_image = np.concatenate([np.asarray((img+1)/2) for img in [pixel_values.clone().cpu().squeeze(1)]])#(b,3,h,w)
                                ref_image = np.concatenate([np.asarray((img+1)/2) for img in [batch["pixel_values_ref"].clone().cpu().squeeze(1)]])#(b,3,h,w)
                                mask = 255 * np.concatenate([np.asarray(img) for img in [batch["mask"].clone().cpu().repeat(1,3,1,1)]])#(b,1,h,w)
                                instance_mask = 255 * np.concatenate([np.asarray(img) for img in [batch["instance_mask"].clone().cpu().repeat(1,3,1,1)]])#(b,1,h,w)
                                if mask_noise:
                                    dilated_mask_log = 255 * np.concatenate([np.asarray(img) for img in [dilated_mask.clone().cpu().repeat(1,3,1,1)]])#(b,1,h,w)
                                    tracker.writer.add_images("train_dilatedmask", dilated_mask_log, global_step, dataformats="NCHW")
                                mask_ref = 255 *  np.concatenate([np.asarray(img) for img in [batch["mask_ref"].clone().cpu().repeat(1,3,1,1)]])#(b,1,h,w)
                                tracker.writer.add_images("train_imagemask", mask, global_step, dataformats="NCHW")
                                tracker.writer.add_images("train_instancemask", instance_mask, global_step, dataformats="NCHW")
                                tracker.writer.add_images("train_mask_ref", mask_ref, global_step, dataformats="NCHW")
                                tracker.writer.add_images("train_input_image", input_image, global_step, dataformats="NCHW")
                                tracker.writer.add_images("train_ref_image", ref_image, global_step, dataformats="NCHW")

                # Cross Attention Loss
                if (args.lambda_attention != 0 or args.lambda_sattention != 0) and phase3_training:
                    attn_loss = 0
                    sattn_loss = 0
                    GT_masks = F.interpolate(input=batch["instance_mask"], size=(16, 16))#(b,1,16,16)
                    for batch_idx in range(args.train_batch_size):
                        curr_cond_batch_idx = args.train_batch_size + batch_idx # instance+prior, not same with break-a-scene where prior+instance

                        if args.lambda_attention != 0:
                            agg_attn = aggregate_attention(controller, res=res_max, from_where=("up", "down"), is_cross=True, select=batch_idx, batch_size=args.train_batch_size)

                            asset_idx = None
                            #print(tokenizer.decode(batch["input_ids"][curr_cond_batch_idx], skip_special_tokens=True))
                            for token_id in modifier_token_id:
                                if token_id in batch["input_ids"][curr_cond_batch_idx]:
                                    asset_idx = (batch["input_ids"][curr_cond_batch_idx] == token_id).nonzero().item()
                                    #print(tokenizer.decode(token_id, skip_special_tokens=True), tokenizer.decode(batch["input_ids"][curr_cond_batch_idx], skip_special_tokens=True), asset_idx if asset_idx is not None else -1)
                            assert asset_idx is not None

                            asset_attn_mask = agg_attn[..., asset_idx]
                            asset_attn_mask = (
                                    asset_attn_mask / asset_attn_mask.max()
                                )
                            attn_loss += F.mse_loss(
                                GT_masks[curr_cond_batch_idx, 0].float(),
                                asset_attn_mask.float(),
                                reduction="mean",
                            )
                            
                            controller.cur_step = 1
                            if global_step % args.training_log_steps == 0:
                                last_sentence = batch["input_ids"][curr_cond_batch_idx]
                                last_sentence = last_sentence[
                                    (last_sentence != 0)
                                    & (last_sentence != 49406)
                                    & (last_sentence != 49407)
                                ]
                                last_sentence = tokenizer.decode(last_sentence)

                                save_cross_attention_vis(
                                    tokenizer,
                                    last_sentence,
                                    attention_maps=agg_attn.detach().cpu(),
                                    path=os.path.join(
                                        img_logs_path, f"attn_{global_step:05}_step.jpg"
                                    ),
                                )
                                os.makedirs(os.path.join(img_logs_path,  'gt_mask'), exist_ok=True)
                                os.makedirs(os.path.join(img_logs_path,  'cross_attn_mask'), exist_ok=True)
                                torchvision.utils.save_image(batch["instance_mask"][curr_cond_batch_idx].unsqueeze(0),  os.path.join(img_logs_path,  'gt_mask', f"gt_mask_{global_step}_{batch_idx}.png"))
                                torchvision.utils.save_image(asset_attn_mask.unsqueeze(0).unsqueeze(0),  os.path.join(img_logs_path, 'cross_attn_mask', f"cross_attn_mask_{global_step}_{batch_idx}.png"))
                            attn_loss = args.lambda_attention * (
                                attn_loss / args.train_batch_size
                            )
                            logs["attn_loss"] = attn_loss.detach().item()
                            loss += attn_loss

                        if args.lambda_sattention != 0:
                            # res attention
                            agg_res_attn = aggregate_attention(controller_res, res=res_max, from_where=("mid", "up"), is_cross=False, select=batch_idx, batch_size=args.train_batch_size)

                            sattn_loss += saloss(
                                agg_res_attn.float(), #[h, w, num_pixels_ref]
                                torch.Tensor([1]).to(accelerator.device),
                                batch["instance_mask"][curr_cond_batch_idx],
                                batch["mask_ref"][curr_cond_batch_idx],
                                reduction="mean",
                                step=global_step,
                                output_dir=args.output_dir,
                                enable_log=global_step % args.training_log_steps == 0,
                            )
                            # resattn logs
                            controller_res.cur_step = 1

                            os.makedirs(os.path.join(img_logs_path,  'input_ref'), exist_ok=True)
                            os.makedirs(os.path.join(img_logs_path,  'input_ref_mask'), exist_ok=True)
                            os.makedirs(os.path.join(img_logs_path,  'sattn'), exist_ok=True)

                            if global_step % args.training_log_steps == 0:
                                save_self_attention_vis_on_pixel(
                                    agg_res_attn.detach().cpu(),
                                    batch["pixel_values"][curr_cond_batch_idx].detach().cpu(),
                                    batch["pixel_values_ref"][curr_cond_batch_idx].detach().cpu(),
                                    path=os.path.join(
                                        img_logs_path, "sattn", f"self2res_attn_{global_step:05}_step.jpg"
                                    )
                                )
                                pixels = torch.cat([batch["pixel_values"][curr_cond_batch_idx].unsqueeze(0), batch["pixel_values_ref"][curr_cond_batch_idx]], dim=0)#(num_ref+1, c, h, w)
                                masks = torch.cat([batch["instance_mask"][curr_cond_batch_idx], batch["mask_ref"][curr_cond_batch_idx]], dim=0).unsqueeze(1)#(num_ref+1, 1, h, w)
                                torchvision.utils.save_image((pixels+1)/2, os.path.join(img_logs_path, "input_ref", f"input_ref_{global_step:05}_step.jpg"))
                                torchvision.utils.save_image(masks, os.path.join(img_logs_path, "input_ref_mask", f"mask_input_ref_{global_step:05}_step.jpg"))
                            sattn_loss = args.lambda_sattention * (
                                sattn_loss / args.train_batch_size
                            )
                            logs["sattn_loss"] = sattn_loss.detach().item()
                            loss += sattn_loss    
                    
                accelerator.backward(loss)

                # No need to keep the attention store
                if phase3_training:
                    controller.attention_store = {}
                    controller.cur_step = 0

                    controller_res.attention_store = {}
                    controller_res.cur_step = 0

                # # Zero out the gradients for all token embeddings except the newly added
                # # embeddings for the concept, as we only want to optimize the concept embeddings
                # if args.modifier_token is not None:
                #     if accelerator.num_processes > 1:
                #         grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                #     else:
                #         grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                #     # Get the index for tokens that we want to zero the grads for
                #     index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                #     for i in range(len(modifier_token_id[1:])):
                #         index_grads_to_zero = index_grads_to_zero & (
                #             torch.arange(len(tokenizer)) != modifier_token_id[i+1]
                #         )
                    
                #     if use_gray:
                #         grads_text_encoder.data[index_grads_to_zero, :int((1 - args.gray_rate)*768)] = grads_text_encoder.data[
                #             index_grads_to_zero, : int((1 - args.gray_rate)*768)
                #         ].fill_(0)
                #     else:
                #         grads_text_encoder.data[index_grads_to_zero, : ] = grads_text_encoder.data[
                #             index_grads_to_zero, :
                #         ].fill_(0)

                if accelerator.sync_gradients:
                    iterables_to_grad_clip = []
                    if phase2_training:
                        iterables_to_grad_clip.extend([text_encoder.parameters(), custom_diffusion_layers.parameters()] \
                                        if args.modifier_token is not None
                                        else [custom_diffusion_layers.parameters()])
                    else:
                        iterables_to_grad_clip.extend([text_encoder.parameters()] if args.modifier_token is not None else [])
                    
                    params_to_clip = (itertools.chain(*iterables_to_grad_clip))
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if args.modifier_token is not None:
                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    index_no_updates[min(modifier_token_id) : max(modifier_token_id) + 1] = False
                    assert max(modifier_token_id) == min(modifier_token_id)
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]
                        # if use_gray and args.gray_rate != 0.:
                        #     accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        #         -1, :int((1 - args.gray_rate)*768)
                        #     ] = cur_embeds_params[-1, :int((1 - args.gray_rate)*768)]
            
                if use_reference_encoder and phase3_training:
                    reference_control_writer.clear()
                    reference_control_reader.clear()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs.update({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})

            # adding logs about tokens structure
            with torch.no_grad():
                modifier_token_embed = {}
                initializer_token_embed = {}
                token_embeds_ = text_encoder.get_input_embeddings().weight.data
                for x, y in zip(modifier_token_id, initializer_token_id):
                    text_modifier, text_init = tokenizer.decode(x).replace("<", "").replace(">", ""), tokenizer.decode(y).replace("<", "").replace(">", "")
                    token_embed_modifier, token_embed_init = token_embeds_[x], token_embeds_[y]
                    modifier_token_embed.update({text_modifier:token_embed_modifier})
                    initializer_token_embed.update({text_init:token_embed_init})
                    logs.update({f'dis_{text_modifier}_{text_init}':F.cosine_similarity(token_embed_modifier, token_embed_init, dim=0).item()})
                modifiers = list(modifier_token_embed.keys())
                initializers = list(initializer_token_embed.keys())
                if len(modifiers) > 1 and len(initializers) > 1:
                    logs.update({f'dis_{modifiers[0]}_{modifiers[1]}':F.cosine_similarity(modifier_token_embed[modifiers[0]], modifier_token_embed[modifiers[1]], dim=0).item()})
                    logs.update({f'dis_{initializers[0]}_{initializers[1]}':F.cosine_similarity(initializer_token_embed[initializers[0]], initializer_token_embed[initializers[1]], dim=0).item()})
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                images = []
                if (args.validation_prompt is not None and global_step % args.validation_steps == 0) or global_step < 2:
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )

                    for k,v in mask_hook.step_store.items():
                        print(f'before attention cal:{k}:{len(v)}')
                    if mask_hook is not None:
                        mask_hook.reset()
                        print(f'reset mask_hook')
                    for k,v in mask_hook.step_store.items():
                        print(f'after attention cal:{k}:{len(v)}')

                    # create pipeline
                    pipeline = EQDIFFPipeline(
                        vae=vae, 
                        text_encoder=accelerator.unwrap_model(text_encoder), 
                        tokenizer=tokenizer, 
                        unet=accelerator.unwrap_model(unet), 
                        #scheduler=DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
                        scheduler=DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
                        reference_encoder=accelerator.unwrap_model(reference_encoder) if use_reference_encoder and phase3_training else None, 
                        controlnet=None,
                    ).to(accelerator.device)

                    # run inference
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    images, ref_image_tensors, images_ref = [], [], []

                    for _ in range(args.num_validation_images):
                        pipeline_output = pipeline(args.validation_prompt, 
                                video_length=1,
                                generator=generator, 
                                fusion_blocks = ref_encoder_addition_kwargs['fusion_blocks'],
                                num_objects=validation_num_objects,
                                use_reference_encoder = True if use_reference_encoder and phase3_training else None, 
                                separate_v=separate_v,
                                placeholder_token=args.modifier_token[0],
                                high_freq_percentage=args.high_freq_percentage,
                                low_freq_percentage=args.low_freq_percentage,
                                freq_embeddings=freq_embeddings,
                                **args_addition['validation_data'],
                                )
                        images.append(pipeline_output.image)
                        if use_reference_encoder and phase3_training: 
                            images_ref.append(pipeline_output.image_ref)
                        ref_image_tensors.append(pipeline_output.ref_image_tensor)

                        # for k,v in mask_hook.step_store.items():
                        #     print(f'attention cal:{k}:{len(v)}')

                        if mask_hook is not None:
                            resize_and_average = lambda tensors_list, h1, w1: torch.stack([torch.nn.functional.interpolate(x, size=(h1, w1), \
                                mode='bilinear', align_corners=False) for x in tensors_list if x.shape[0]==2]).mean(dim=0)
                            for k, v in mask_hook.step_store.items():
                                if len(v) > 0:
                                    # for vi in v:
                                    #     print(vi.shape)
                                    mask_save = resize_and_average(v, 128, 128)
                                    mask_path = os.path.join(args.output_dir, 'mask_validation')
                                    os.makedirs(mask_path, exist_ok=True)
                                    save_name = os.path.join(mask_path, f'{global_step:04}_{k}.png')
                                    save_image(mask_save, save_name)

                    # Attention Map visualization
                    if use_reference_encoder and phase3_training:
                        controller.cur_step = 1
                        full_agg_attn = aggregate_attention(controller, 
                                res=res_max, from_where=("up", "down"), is_cross=True, select=0, batch_size=1
                            )
                        save_cross_attention_vis(
                            tokenizer,
                            args.validation_prompt,
                            attention_maps=full_agg_attn.detach().cpu(),
                            path=os.path.join(
                                args.output_dir, f"{global_step:05}_full_attn.jpg"
                            ),
                        )
                        controller.cur_step = 0
                        controller.attention_store = {}

                        # for res attention visualization
                        controller_res.cur_step = 1
                        full_agg_res_attn = aggregate_attention(controller_res,
                                res=res_max, from_where=("up", "mid"), is_cross=False, select=0, batch_size=1
                            )
                        save_self_attention_vis_on_pixel(
                                full_agg_res_attn.detach().cpu(),
                                images[-1][0].detach().cpu(),
                                ref_image_tensors[-1].detach().cpu(),
                                path=os.path.join(
                                    args.output_dir, f"{global_step:05}_full_res_attn.jpg"
                                ),
                        )
                        # save_self_attention_vis(
                        #     attention_maps=full_agg_res_attn.detach().cpu(),
                        #     path=os.path.join(
                        #         args.output_dir, f"{global_step:05}_full_res_attn.jpg"
                        #     ),
                        # )
                        controller_res.cur_step = 0
                        controller_res.attention_store = {}

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.concatenate([np.asarray(img) for img in images])#(b,3,h,w)
                            ref_images = np.concatenate([np.asarray(img.clone().cpu()) for img in ref_image_tensors])#(b,3,h,w)
                            tracker.writer.add_images("validation", np_images, global_step, dataformats="NCHW")
                            tracker.writer.add_images("validation_ref", ref_images, global_step, dataformats="NCHW")
                            if use_reference_encoder and phase3_training:
                                np_images_ref = np.concatenate([np.asarray(img) for img in images_ref])#(b,3,h,w)
                                tracker.writer.add_images("validation_ref_style", np_images_ref, global_step, dataformats="NCHW")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

                    # restore the reference encoder and unet internal state after validation
                    # since the classifier_free is set to True in validation and 
                    # num objects is not the same in train and validation
                    if use_reference_encoder and phase3_training:
                        reference_control_writer = ReferenceAttentionControl(reference_encoder, 
                                                                             do_classifier_free_guidance=False, 
                                                                             mode='write', 
                                                                             fusion_blocks=ref_encoder_addition_kwargs['fusion_blocks'])
                        reference_control_reader = ReferenceAttentionControl(unet, 
                                                                             do_classifier_free_guidance=False, 
                                                                             mode='read', 
                                                                             fusion_blocks=ref_encoder_addition_kwargs['fusion_blocks'], num_objects=num_objects_given)
            if use_reference_encoder and phase3_training:
                num_objects_given_prev = num_objects_given
    
    # Save the custom diffusion layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = EQDIFFPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae, 
                    text_encoder=accelerator.unwrap_model(text_encoder), 
                    tokenizer=tokenizer, 
                    unet=accelerator.unwrap_model(unet), 
                    reference_encoder=reference_encoder if (use_reference_encoder and phase3_training) else None, 
                )
        print(f'(use_reference_encoder and phase3_training):{(use_reference_encoder and phase3_training)}')
        pipeline.save_pretrained(args.output_dir)

        accelerator.unwrap_model(unet).to(torch.float32).save_attn_procs(args.output_dir, safe_serialization=False)

        # save woref model
        output_dir_woref = os.path.join(args.output_dir, 'woref')
        os.makedirs(output_dir_woref, exist_ok=True)
        pipeline = EQDIFFPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae, 
                    text_encoder=accelerator.unwrap_model(text_encoder), 
                    tokenizer=tokenizer, 
                    unet=accelerator.unwrap_model(unet), 
                    reference_encoder=None, 
                )
        pipeline.save_pretrained(output_dir_woref)
        accelerator.unwrap_model(unet).to(torch.float32).save_attn_procs(output_dir_woref, safe_serialization=False)
        save_new_embed(
            text_encoder,
            modifier_token_id,
            accelerator,
            args,
            args.output_dir,
            safe_serialization=False,
        )
        torch.save(freq_embeddings.state_dict(), os.path.join(args.output_dir, 'freq_embeddings.pth'))

        # Final inference
        # Load previous pipeline
        final_inference = True
        if final_inference:
            unet = UNet3DConditionModel.from_pretrained_2d(args.output_dir, subfolder="unet", unet_additional_kwargs=unet_additional_kwargs)
            pipeline = EQDIFFPipeline.from_pretrained(
                        args.output_dir,
                        unet=unet,
                    ).to(accelerator.device)
            pipeline.scheduler = DDIMScheduler.from_pretrained(args.output_dir, subfolder="scheduler") 

            # load attention processors
            weight_name = (
                "pytorch_custom_diffusion_weights.bin"
            )
            pipeline.unet.load_attn_procs(args.output_dir, weight_name=weight_name, mask_hook=mask_hook)
            # for token in args.modifier_token:
            #     token_weight_name =  f"{token}.bin"
            #     pipeline.load_textual_inversion(args.output_dir, weight_name=token_weight_name)

            loaded_freq_embeddings = LearnableEmbeddings(2, embedding_dim)
            loaded_freq_embeddings.load_state_dict(torch.load(os.path.join(args.output_dir, 'freq_embeddings.pth')))
            print(loaded_freq_embeddings.embedding.weight == freq_embeddings.embedding.weight)

            
            # run inference
            if args.validation_prompt and args.num_validation_images > 0:
                seed = 247247923479279
                generator = torch.Generator(device='cuda')
                generator.manual_seed(seed)
                images = [
                    pipeline(args.validation_prompt, 
                            video_length=1,
                            generator=generator, 
                            fusion_blocks = ref_encoder_addition_kwargs['fusion_blocks'],
                            num_objects=validation_num_objects,
                            freq_embeddings=loaded_freq_embeddings,
                            **args_addition['validation_data'],
                            ).image
                    for _ in range(args.num_validation_images)
                ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.concatenate([np.asarray(img) for img in images])
                        tracker.writer.add_images("test", np_images, epoch, dataformats="NCHW")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "test": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

    accelerator.end_training()


def get_average_attention(controller):
        # for k, v in controller.attention_store.items():
        #     for vi in v:
        #         print(f'{k}:{vi.shape},{len(v)}')
        average_attention = {
            key: [
                item / controller.cur_step
                for item in controller.attention_store[key]
            ]
            for key in controller.attention_store
        }
        return average_attention

def aggregate_attention(
    controller, res: int, from_where: List[str], is_cross: bool, select: int, batch_size: int,
):
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(
                    batch_size, -1, res, res, item.shape[-1]
                )[select]
                #print(f'cross_maps:{cross_maps.sum([3])}')
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    #print(f'out:{out.sum(2)}')
    return out

@torch.no_grad()
def save_cross_attention_vis(tokenizer, prompt, attention_maps, path):
    tokens = tokenizer.encode(prompt)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(
            image, tokenizer.decode(int(tokens[i]))
        )
        images.append(image)
    vis = ptp_utils.view_images(np.stack(images, axis=0))
    vis.save(path)

@torch.no_grad()
def save_self_attention_vis(attention_maps, path):
    attn_maps = attention_maps
    # attention_maps: [h, w, num_pixels_ref]
    num_vis = 16
    h, w, l = attention_maps.shape
    num_ref = int(l//h//w)

    attention_maps = attention_maps.reshape(h*w, num_ref, h, w)
    #print(f'attention_maps_res:{attention_maps.sum([1,2,3])}')
    #print(f'attention_maps_res:{attention_maps.var([1,2,3])}')
    attention_maps = rearrange(attention_maps, "l n a b -> l a (n b)")
    select_idx = (torch.arange(0, num_vis) * ((h*w-1)/num_vis)).int().tolist()
    images = attention_maps[select_idx,...]#(num_select, h, n*w)
    images = F.interpolate(images.unsqueeze(1), scale_factor=2, mode='bilinear')
    T.save_image(images, path)

    # vis from res perspect
    select_idx_2 = (torch.arange(0, num_vis) * ((l-1)/num_vis)).int().tolist()
    attn_maps_2 = attn_maps[:,:,select_idx_2].permute(2,0,1)#(num_select, h, w, )
    attn_maps_2 = attn_maps_2 / attn_maps_2.max(dim=0, keepdim=True).values
    images = F.interpolate(attn_maps_2.unsqueeze(1), scale_factor=2, mode='bilinear')
    T.save_image(images, path.replace('.jpg', '_2.jpg'))

@torch.no_grad()
def save_self_attention_vis_on_pixel(attention_maps, image, image_ref, path):
    # attention_maps: [h, w, num_pixels_ref], and num_pixels_ref = num_ref*h*w
    # image: [c, h1, w1,]
    # image_ref: [num_ref, c, h1, w1]

    attn_maps = attention_maps
    h, w, l = attention_maps.shape
    num_ref = int(l // h // w)
    hv, wv = 256, 256

    
    image_ref = F.interpolate(image_ref, (hv, wv))
    image_ref = rearrange(image_ref, "n c h w -> c h (n w)")

    attention_maps = attention_maps.reshape(h * w, num_ref, h, w)
    attention_maps = rearrange(attention_maps, "l n a b -> l a (n b)")

    # define the sampling idx
    horizontal_step, vertical_step = 4, 4
    base_idx = list(range(1, w, horizontal_step))
    select_idx = []
    for i in range(1, h, vertical_step):
        select_idx += [idx+i*w for idx in base_idx]
        

    # # Step 1: 在image上为select_idx中的每个点绘制可变大小的边框
    # image = F.interpolate(image.unsqueeze(0), (h, w))[0]
    # self_wi_bbox = []
    # for idx in select_idx:
    #     h_idx, w_idx = divmod(idx, w)
    #     # print(f'idx:{idx}, {h_idx}, {w_idx}')
    #     # 定义bbox的大小（此处为示例，您可以根据需要调整）
    #     bbox_size = 1
        
    #     # 使用cv2.rectangle绘制边框
    #     image_np = TF.to_pil_image(image)
    #     image_np = np.array(image_np).copy()
    #     cv2.rectangle(image_np, (w_idx - bbox_size, h_idx - bbox_size), 
    #                   (w_idx + bbox_size, h_idx + bbox_size), (255, 0, 0), 1)  # 红色边框

    #     image_tensor = TF.to_tensor(image_np)
    #     self_wi_bbox.append(image_tensor)
    # self_wi_bbox = torch.stack(self_wi_bbox, dim=0)#[num_vis, 3, h, w]
    # self_wi_bbox = F.interpolate(self_wi_bbox, (hv, wv), mode='bilinear')
    # #save_image(self_wi_bbox, 'self_wi_bbox.jpg')

    # Step 2: 
    image_after = F.interpolate(image.unsqueeze(0), (hv, wv))[0]
    X, Y = torch.meshgrid(torch.arange(1, w, horizontal_step), torch.arange(1, h, vertical_step), indexing='xy')
    X, Y = torch.clamp(X * wv / w, 0, wv-1), torch.clamp(Y * hv / h, 0, hv-1)
    select_idx_after = (X + Y*wv).flatten()
    # print(select_idx_after)
    select_idx_after = select_idx_after.to(torch.int32).tolist()
    self_wi_bbox_after = []
    for idx in select_idx_after:
        h_idx, w_idx = divmod(idx, wv)
        # print(f'idx:{idx}, {h_idx}, {w_idx}')
        # 定义bbox的大小（此处为示例，您可以根据需要调整）
        bbox_size = 5
        
        # 使用cv2.rectangle绘制边框
        image_np = TF.to_pil_image(image_after)
        image_np = np.array(image_np).copy()
        # print((w_idx - bbox_size, h_idx - bbox_size), (w_idx + bbox_size, h_idx + bbox_size))
        cv2.rectangle(image_np, (w_idx - bbox_size, h_idx - bbox_size), 
                      (w_idx + bbox_size, h_idx + bbox_size), (255, 0, 0), 2)  # 红色边框

        image_tensor = TF.to_tensor(image_np)
        #print(image.shape)
        self_wi_bbox_after.append(image_tensor)
    self_wi_bbox_after = torch.stack(self_wi_bbox_after, dim=0)#[num_vis, 3, h, w]
    self_wi_bbox_after = F.interpolate(self_wi_bbox_after, (hv, wv), mode='bilinear')
    # save_image(self_wi_bbox_after, 'self_wi_bbox_after.jpg')

    # Step 2: 处理每个select_idx对应的attention_map，并叠加在image_ref上
    self2res_attn_vis = []
    p = 0.3
    for idx in select_idx:
        h_idx, w_idx = divmod(idx, w)
        attention_map = attention_maps[idx,...]  # [h, num_ref*w]
        #print(f'attention_map:{attention_map.shape}')
        
        # 将attention_map的值映射到colormap的值
        attention_map = attention_map/ (attention_map.max(dim=0).values.max(dim=0).values)  # 归一化到 [0, 1]
        
        colormap = plt.get_cmap('viridis')
        attention_map_colormap = colormap(attention_map.numpy())[:,:,:3]  # 忽略alpha通道 [h, num_ref*w, 3]
        attention_map_colormap = torch.from_numpy(attention_map_colormap.transpose((2, 0, 1)))  # 调整为PyTorch格式 [3, h, num_ref*w]

        attention_map_colormap = rearrange(attention_map_colormap, 'c h (n w) -> n c h w', n=num_ref, w=w)
        attention_map_colormap = F.interpolate(attention_map_colormap, (hv, wv), mode="bilinear")
        attention_map_colormap = rearrange(attention_map_colormap, 'n c h w -> c h (n w)')
        
        #attention_map_colormap = F.interpolate(attention_map_colormap.unsqueeze(0), scale_factor=2, mode='bilinear')[0]# [3, h, num_ref*w]
        self2res_attn_vis.append(image_ref*p + attention_map_colormap*(1-p))
    self2res_attn_vis = torch.stack(self2res_attn_vis, dim=0)#(num_vis, 3, h, num_ref*w)

    # Step 3: 将画红框标注和attention_map结果concat到一起并保存
    final_visualization = torch.cat([self_wi_bbox_after, self2res_attn_vis], dim=3)
    save_image(final_visualization, path)

if __name__ == "__main__":
    args = parse_args()
    main(args)