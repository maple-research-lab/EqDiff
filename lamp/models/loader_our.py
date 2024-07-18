# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
from collections import defaultdict
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import requests
import safetensors
import torch
from huggingface_hub import hf_hub_download, model_info
from torch import nn

from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta
from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    _get_model_file,
    deprecate,
    is_accelerate_available,
    is_omegaconf_available,
    is_transformers_available,
    logging,
)
from diffusers.utils.import_utils import BACKENDS_MAPPING

from lamp.models.attention_processor_custom import CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor


if is_transformers_available():
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

TEXT_INVERSION_NAME = "learned_embeds.bin"
TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"

CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"

def text_encoder_attn_modules(text_encoder):
    attn_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


def text_encoder_mlp_modules(text_encoder):
    mlp_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            mlp_mod = layer.mlp
            name = f"text_model.encoder.layers.{i}.mlp"
            mlp_modules.append((name, mlp_mod))
    else:
        raise ValueError(f"do not know how to get mlp modules for: {text_encoder.__class__.__name__}")

    return mlp_modules


def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k

            raise ValueError(
                f"There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}."
            )

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class UNet2DConditionLoadersMixin_our:
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], mask_hook:Optional[None], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
        and be a `torch.nn.Module` class.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """
        # from diffusers.models.attention_processor import (
        #     CustomDiffusionAttnProcessor,
        # )
        
        from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear, LoRAConv2dLayer, LoRALinearLayer

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alphas = kwargs.pop("network_alphas", None)

        _pipeline = kwargs.pop("_pipeline", None)

        is_network_alphas_none = network_alphas is None

        allow_pickle = False

        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        lora_layers_list = []

        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

        if is_lora:
            # correct keys
            state_dict, network_alphas = self.convert_state_dict_legacy_attn_format(state_dict, network_alphas)

            if network_alphas is not None:
                network_alphas_keys = list(network_alphas.keys())
                used_network_alphas_keys = set()

            lora_grouped_dict = defaultdict(dict)
            mapped_network_alphas = {}

            all_keys = list(state_dict.keys())
            for key in all_keys:
                value = state_dict.pop(key)
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

                # Create another `mapped_network_alphas` dictionary so that we can properly map them.
                if network_alphas is not None:
                    for k in network_alphas_keys:
                        if k.replace(".alpha", "") in key:
                            mapped_network_alphas.update({attn_processor_key: network_alphas.get(k)})
                            used_network_alphas_keys.add(k)

            if not is_network_alphas_none:
                if len(set(network_alphas_keys) - used_network_alphas_keys) > 0:
                    raise ValueError(
                        f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                    )

            if len(state_dict) > 0:
                raise ValueError(
                    f"The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
                )

            for key, value_dict in lora_grouped_dict.items():
                attn_processor = self
                for sub_key in key.split("."):
                    attn_processor = getattr(attn_processor, sub_key)

                # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
                # or add_{k,v,q,out_proj}_proj_lora layers.
                rank = value_dict["lora.down.weight"].shape[0]

                if isinstance(attn_processor, LoRACompatibleConv):
                    in_features = attn_processor.in_channels
                    out_features = attn_processor.out_channels
                    kernel_size = attn_processor.kernel_size

                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        lora = LoRAConv2dLayer(
                            in_features=in_features,
                            out_features=out_features,
                            rank=rank,
                            kernel_size=kernel_size,
                            stride=attn_processor.stride,
                            padding=attn_processor.padding,
                            network_alpha=mapped_network_alphas.get(key),
                        )
                elif isinstance(attn_processor, LoRACompatibleLinear):
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        lora = LoRALinearLayer(
                            attn_processor.in_features,
                            attn_processor.out_features,
                            rank,
                            mapped_network_alphas.get(key),
                        )
                else:
                    raise ValueError(f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module.")

                value_dict = {k.replace("lora.", ""): v for k, v in value_dict.items()}
                lora_layers_list.append((attn_processor, lora))

                if low_cpu_mem_usage:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(lora, value_dict, device=device, dtype=dtype)
                else:
                    lora.load_state_dict(value_dict)

        elif is_custom_diffusion:
            attn_processors = {}
            custom_diffusion_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                if len(value) == 0:
                    custom_diffusion_grouped_dict[key] = {}
                else:
                    if "to_out" in key:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                    else:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                    custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in custom_diffusion_grouped_dict.items():

                if len(value_dict) == 0:# the custom_diffusion_attn_processor does not include trainable params
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=False, train_q_out=False, train_res=False, hidden_size=None, cross_attention_dim=None
                    )
                else:
                    if key.endswith('attn2.processor'):
                        cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                        hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                        train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                        attn_processors[key] = CustomDiffusionAttnProcessor(
                            train_kv=True,
                            train_q_out=train_q_out,
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                        )
                        attn_processors[key].load_state_dict(value_dict)
                    elif key.endswith('attn1.processor'):
                        if key.startswith("mid_block"):
                            place_in_unet = 'mid'
                        elif key.startswith("up_block"):
                            place_in_unet = 'up'
                        elif key.startswith("down_blocks"):
                            place_in_unet = 'down'
                        else:
                            raise ValueError(f'unexpected name')
                        cross_attention_dim = None
                        hidden_size = value_dict["to_k_res.weight"].shape[0]
                        train_res = True if "to_k_res.weight" in value_dict else False
                        attn_processors[key] = CustomDiffusionAttnProcessor(
                            train_kv=False,
                            train_q_out=False,
                            train_res=train_res,
                            use_integra_ref_mask=True,
                            mask_hook=mask_hook,
                            place_in_unet=place_in_unet,
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                        )
                        attn_processors[key].load_state_dict(value_dict)

        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )

        # <Unsafe code
        # We can be sure that the following works as it just sets attention processors, lora layers and puts all in the same dtype
        # Now we remove any existing hooks to
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False
        if _pipeline is not None:
            for _, component in _pipeline.components.items():
                if isinstance(component, nn.Module):
                    if hasattr(component, "_hf_hook"):
                        is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                        is_sequential_cpu_offload = isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)
                        logger.info(
                            "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                        )
                        remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

        # only custom diffusion needs to set attn processors
        if is_custom_diffusion:
            self.set_attn_processor(attn_processors)

        # set lora layers
        for target_module, lora_layer in lora_layers_list:
            target_module.set_lora_layer(lora_layer)

        self.to(dtype=self.dtype, device=self.device)

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />

    def convert_state_dict_legacy_attn_format(self, state_dict, network_alphas):
        is_new_lora_format = all(
            key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
        )
        if is_new_lora_format:
            # Strip the `"unet"` prefix.
            is_text_encoder_present = any(key.startswith(self.text_encoder_name) for key in state_dict.keys())
            if is_text_encoder_present:
                warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
                logger.warn(warn_message)
            unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
            state_dict = {k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

        # change processor format to 'pure' LoRACompatibleLinear format
        if any("processor" in k.split(".") for k in state_dict.keys()):

            def format_to_lora_compatible(key):
                if "processor" not in key.split("."):
                    return key
                return key.replace(".processor", "").replace("to_out_lora", "to_out.0.lora").replace("_lora", ".lora")

            state_dict = {format_to_lora_compatible(k): v for k, v in state_dict.items()}

            if network_alphas is not None:
                network_alphas = {format_to_lora_compatible(k): v for k, v in network_alphas.items()}
        return state_dict, network_alphas

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        r"""
        Save an attention processor to a directory so that it can be reloaded using the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        """
        # from .models.attention_processor import (
        #     CustomDiffusionAttnProcessor,
        #     CustomDiffusionXFormersAttnProcessor,
        # )

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        is_custom_diffusion = any(
            isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
                }
            )
            state_dict = model_to_save.state_dict()
            for name, attn in self.attn_processors.items():
                if len(attn.state_dict()) == 0:
                    state_dict[name] = {}
        else:
            model_to_save = AttnProcsLayers(self.attn_processors)
            state_dict = model_to_save.state_dict()

        if weight_name is None:
            if safe_serialization:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else LORA_WEIGHT_NAME

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    def fuse_lora(self, lora_scale=1.0):
        self.lora_scale = lora_scale
        self.apply(self._fuse_lora_apply)

    def _fuse_lora_apply(self, module):
        if hasattr(module, "_fuse_lora"):
            module._fuse_lora(self.lora_scale)

    def unfuse_lora(self):
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        if hasattr(module, "_unfuse_lora"):
            module._unfuse_lora()
