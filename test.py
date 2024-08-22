import argparse

import os
import os.path as osp
from tqdm import tqdm
import torch
import numpy as np
from torchvision.utils import save_image
from transformers import CLIPTextModel, CLIPTokenizer

from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)

from eqdiff.models.unet import UNet3DConditionModel
from eqdiff.models.align_hook import MasksHook
from eqdiff.util import save_videos_grid, ddim_inversion, load_weights_into_unet
from eqdiff.pipelines.pipeline_eqdiff_multi_inj_refnew_difftextemb2 import EQDIFFPipeline
from eqdiff.tools import *

live_list = [
    "cat",
    "cat2",
    "dog",
    "dog2",
    "dog3",
    "dog5",
    "dog6",
    "dog7",
    "dog8",
]

no_live_list = [
  "cat_toy",
  "backpack",
  "backpack_dog",
  "bear_plushie",
  "berry_bowl",
  "can",
  "candle",
  "clock",
  "colorful_sneaker",
  "duck_toy",
  "fancy_boot",
  "grey_sloth_plushie",
  "monster_toy",
  "pink_sunglasses",
  "poop_emoji",
  "rc_car",
  "red_cartoon",
  "robot_toy",
  "shiny_sneaker",
  "teapot",
  "vase",
  "wolf_plushie"
]

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Residual Reference Attention testing script.")
    parser.add_argument("--DATASET_NAME", type=str, default="cat_toy")
    parser.add_argument("--PROMPT_NAME", type=str, default="cat toy")
    parser.add_argument("--PROMPTS_FILE", type=str, default="./prompts/base_prompt.txt")
    parser.add_argument("--PATH_PREFIX", type=str, default="three_stage_wi_attn_loss_trainbothfirstref_phase1step100_phase2step100_phase3step400_phase4step200_woflip_caloss1e-1_scaleinitlr_mix+grayv5_10_0.5_wisa_wimask_")
    parser.add_argument("--MODE", type=str, default="style")
    parser.add_argument("--SD_PATH", type=str, default="/path/to/stable-diffusion-v1-4")
    parser.add_argument("--config", type=str, default="./configs/custom_diffusion.yaml")
    parser.add_argument(
        "--modifier_token",
        type=str,
        default="<new2>",
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--use_reference_encoder", action="store_true", help="if use reference encoder")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    n_iter = 1
    DATASET_NAME=args.DATASET_NAME
    PROMPT_NAME=args.PROMPT_NAME
    
    mode = args.MODE
    if mode == "None":
        if DATASET_NAME in live_list:
            mode_ = "live"
            prompts_file = './prompts/prompt_live.txt'
            
        elif DATASET_NAME in no_live_list:
            mode_ = "nolive"
            prompts_file = './prompts/prompt_nolive.txt'
        else:
            pass
    else:
        assert mode == "style"
        prompts_file = './prompts/prompt_style.txt'

    with open(prompts_file, 'r') as file:
        lines = file.readlines()
    prompts = [line.strip().replace("{}", "<new2> "+DATASET_NAME.replace("_"," ")) for line in lines]

    #ref_prompt=[f"photo of a <new2> {PROMPT_NAME}"]
    #config = OmegaConf.load(f"./configs/reference_diffusion_{DATASET_NAME}.yaml")
    args_addition = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    #ref_img_path = config.validation_data.ref_img_path
    if args.guidance_scale is not None:
        guidance_scale = args.guidance_scale
        args_addition["validation_data"]['guidance_scale'] = guidance_scale
    else:
        guidance_scale = args_addition["validation_data"]['guidance_scale']
    print(f"guidance_scale: {guidance_scale}")
    if mode == "None":
        save_path=f"./outputs/{args.PATH_PREFIX}{DATASET_NAME}/nostyle_{guidance_scale}"
    else:
        save_path=f"./outputs/{args.PATH_PREFIX}{DATASET_NAME}/style_{guidance_scale}"
    pretrained_model_path=f"./checkpoints/{args.PATH_PREFIX}{DATASET_NAME}"
    # save_path=f"./outputs/three_stage_wi_attn_loss_trainbothfirstref_phase1step100_phase2step100_phase3step400_phase4step400_woflip_caloss1e-1_scaleinitlr_mix+grayv5_10_0.5_{DATASET_NAME}"
    # pretrained_model_path = f"./checkpoints/three_stage_wi_attn_loss_trainbothfirstref_phase1step100_phase2step100_phase3step400_phase4step400_woflip_caloss1e-1_scaleinitlr_mix+grayv5_10_0.5_{DATASET_NAME}"
    
    args.modifier_token = args.modifier_token.split("+")
    
    num_objects=1
    p_min, p_max=0., 1.

    # seed = 247247923479279
    seed = 147247923479279
    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)

    os.makedirs(save_path, exist_ok=True)

    mask_hook = MasksHook()
    # load model
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, 
                                                subfolder="unet", 
                                                unet_additional_kwargs=args_addition["unet_additional_kwargs"])
    
    text_encoder = CLIPTextModel.from_pretrained(args.SD_PATH, 
                                                subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.SD_PATH, 
                                                subfolder="tokenizer")
    
    # replace the forward method of the text encoder to inject the word embedding
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    # # with reference 
    pipeline = EQDIFFPipeline.from_pretrained(
                            pretrained_model_path,
                            unet=unet,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            ).to("cuda")
    pipeline.scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler") 

    weight_name = ("pytorch_custom_diffusion_weights.bin")
    pipeline.unet.load_attn_procs(pretrained_model_path, 
                                  weight_name=weight_name,
                                mask_hook=mask_hook)
    pipeline.load_textual_inversion(pretrained_model_path, weight_name="<new2>.bin")

    embedding_dim = text_encoder.get_input_embeddings().weight.data.shape[1]
    loaded_freq_embeddings = LearnableEmbeddings(2, embedding_dim)
    loaded_freq_embeddings.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'freq_embeddings.pth')))

    # inference
    print(f"use_reference_encoder: {args.use_reference_encoder}")
    for i, prompt in enumerate(prompts):
        image = pipeline(
            prompt,
            video_length=1,
            generator=generator,
            fusion_blocks='midup',
            num_objects=num_objects,
            use_reference_encoder=True if args.use_reference_encoder else False,
            separate_v=False,
            placeholder_token=args.modifier_token[0],
            freq_embeddings=loaded_freq_embeddings,
            p_min=p_min,
            p_max=p_max,
            **args_addition["validation_data"],
        ).image
        save_name = prompt.replace(" ", "_")+".png"
        save_image(image, osp.join(save_path, save_name), nrow=1)



if __name__ == "__main__":
    args = parse_args()
    main(args)