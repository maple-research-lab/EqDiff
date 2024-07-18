#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Custom Diffusion authors and the HuggingFace Inc. team. All rights reserved.
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
import inspect
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
import torchvision.utils as TU
from torchvision.utils import save_image
import torchvision
import cv2
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt


import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.fft as fft
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
from transformers import AutoTokenizer, PretrainedConfig
from collections import defaultdict
from omegaconf import OmegaConf


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    input_index = [example["index"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    instance_mask = [example["instance_mask"] for example in examples]
    mask = [example["mask"] for example in examples]

    input_ids_ref = [example["instance_prompt_ref_ids"] for example in examples]
    pixel_values_ref = [example["instance_images_ref"] for example in examples]
    mask_ref = [example["mask_ref"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids = [example["class_prompt_ids"] for example in examples] + input_ids
        input_index = [example["class_index"] for example in examples] + input_index
        pixel_values = [example["class_images"] for example in examples] + pixel_values
        instance_mask = [example["class_mask"] for example in examples] + instance_mask
        mask = [example["class_mask"] for example in examples] + mask

        input_ids_ref = [example["class_prompt_ids_ref"] for example in examples] + input_ids_ref
        pixel_values_ref = [example["class_images_ref"] for example in examples] + pixel_values_ref
        mask_ref = [example["class_mask_ref"] for example in examples] + mask_ref

    input_ids = torch.cat(input_ids, dim=0)#(b, 77)
    index = torch.stack(input_index)#(b)
    pixel_values = torch.stack(pixel_values)#(b, c, h, w)
    instance_mask = torch.stack(instance_mask)#(b, h, w)
    mask = torch.stack(mask)#(b, h, w)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    instance_mask = instance_mask.to(memory_format=torch.contiguous_format).float()/255
    mask = mask.to(memory_format=torch.contiguous_format).float()/255

    input_ids_ref = torch.stack(input_ids_ref, dim=0)#(b, n, 77)
    pixel_values_ref = torch.stack(pixel_values_ref)#(b, n, c, h, w)
    mask_ref = torch.stack(mask_ref)#(b, n, h, w)
    pixel_values_ref = pixel_values_ref.to(memory_format=torch.contiguous_format).float()
    mask_ref = mask_ref.to(memory_format=torch.contiguous_format).float()/255

    batch = {"input_ids": input_ids, 
             "index": index,
             "pixel_values": pixel_values, 
             "instance_mask":instance_mask.unsqueeze(1),
             "mask": mask.unsqueeze(1),
             "input_ids_ref": input_ids_ref, 
             "pixel_values_ref": pixel_values_ref, 
             "mask_ref": mask_ref}
    # for k, v in batch.items():
    #     print(f'{k}:{v.shape}')
    # print(f'{batch["instance_mask"].sum([1,2,3])}')
    # print(f'{batch["mask"].sum([1,2,3])}')
    # print(f'{batch["mask_ref"].sum([1,2,3])}')
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
        num_ref=1,
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug
        self.num_ref = num_ref

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        self.instance_images_dict = defaultdict(dict)
        for id_c, concept in enumerate(concepts_list):
            inst_img_path = [
                (id_c, x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            for x in Path(concept["instance_data_dir"]).iterdir():
                if x.is_file():
                    self.instance_images_dict[id_c].update({x: concept["instance_prompt"]})  
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)
        self.flip_p = 0.5 * hflip

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # augmentation for prior preservation
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.image_transforms_plus = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        elif scale < self.size:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        else:
            instance_image = image
            mask = 1.0 * mask
        return instance_image, mask
    
    def preprocess_wi_mask(self, image, mask, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        if isinstance(mask, torch.Tensor):
            mask = F.interpolate(mask.unsqueeze(0), (scale, scale), mode="nearest")[0][0]#(h,w)
            mask = mask.numpy().astype(np.uint8)

        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        instance_mask = np.zeros((self.size, self.size))
        image_mask = np.zeros((self.size // factor, self.size // factor))
        print(scale, self.size)
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            instance_mask = mask[top : top + inner, left : left + inner]
            image_mask = 255 * np.ones((self.size // factor, self.size // factor))
        elif scale < self.size:
            instance_image[top : top + inner, left : left + inner, :] = image
            instance_mask[top : top + inner, left : left + inner] = mask
            image_mask[
               top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 255.
        else:
            print(f'size match')
            instance_image = image
            instance_mask = mask
            image_mask = torch.ones_like(instance_mask) * 255.
        return instance_image, instance_mask, image_mask

    def __getitem__(self, index):
        # 1. get instance image
        example = {}
        id_c, instance_image_path, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        #instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        #instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)
        
        # 2. get instance mask
        (instance_name, instance_id) = str(instance_image_path).split("/")[-2:]
        instance_mask_path = osp.join(osp.dirname(str(instance_image_path)), 'seg_masks', osp.splitext(instance_id)[0], instance_name.replace('_', ' ') + '.jpg')
        assert osp.exists(instance_mask_path), f'{instance_mask_path} not exists'
        instance_mask = Image.open(instance_mask_path)#(h,w)
        instance_mask = np.array(instance_mask).astype(np.uint8)#(h,w,3)
        instance_mask = torch.Tensor(instance_mask).permute(2,0,1)#(3,h,w)

        # if random.random() < self.flip_p:
        #     instance_image, instance_mask = TF.hflip(instance_image), TF.hflip(instance_mask)

        # disable aug
        instance_image, instance_mask, image_mask = self.preprocess_wi_mask(instance_image, instance_mask, self.size, self.interpolation)
        instance_mask = F.interpolate(torch.Tensor(instance_mask)[None, None, :, :], size=(self.mask_size, self.mask_size), mode='nearest')[0][0].numpy()
        #print(f'instance_mask:{instance_mask.shape}')

        #print(f'instance_prompt:{type(instance_prompt)}, {instance_prompt}')
        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["instance_mask"] = torch.from_numpy(instance_mask)#(h, w)
        example["mask"] = torch.from_numpy(image_mask)#(h, w)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # 2. get ref image in the same concept
        # excluded_instance_images_dict = self.instance_images_dict[id_c].copy()
        # excluded_instance_images_dict.pop(instance_image_path)
        # instance_prompt_ref_samples = random.sample(list(excluded_instance_images_dict.items()), self.num_ref)

        # instance_image_ref_list, instance_prompt_ref_list, mask_ref_list = [], [], []
        # for i, (instance_image_ref, instance_prompt_ref) in enumerate(instance_prompt_ref_samples):
        #     assert (not instance_image_ref == instance_image_path), f"{instance_image_ref} vs {instance_image_path}"

        #     #instance_image_ref, instance_prompt_ref = random.choice(self.instance_images_dict[id_c])
        #     #while (instance_image_ref == instance_image_path):
        #     #    instance_image_ref, instance_prompt_ref = random.choice(self.instance_images_dict[id_c])
        #     #assert (not instance_image_ref == instance_image_path), f"{instance_image_ref} vs {instance_image_path}"
        #     instance_image_ref = Image.open(instance_image_ref)
        #     if not instance_image_ref.mode == "RGB":
        #         instance_image_ref = instance_image_ref.convert("RGB")

        #     instance_image_ref = self.flip(instance_image_ref)
        #     # apply resize augmentation and create a valid image region mask
        #     random_scale_ref = self.size
        #     if self.aug:
        #         random_scale_ref = (
        #             np.random.randint(self.size // 3, self.size + 1)
        #             if np.random.uniform() < 0.66
        #             else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
        #         )
        #     instance_image_ref, mask_ref = self.preprocess(instance_image_ref, random_scale_ref, self.interpolation)
        #     if random_scale_ref < 0.6 * self.size:
        #         instance_prompt_ref = np.random.choice(["a far away ", "very small "]) + instance_prompt_ref
        #     elif random_scale_ref > self.size:
        #         instance_prompt_ref = np.random.choice(["zoomed in ", "close up "]) + instance_prompt_ref

        #     instance_image_ref_list.append(instance_image_ref)
        #     instance_prompt_ref_list.append(instance_prompt_ref)
        #     mask_ref_list.append(mask_ref)
        
        # instance_images_ref = np.stack(instance_image_ref_list, axis=0)#(n, h, w, c)
        # mask_ref = np.stack(mask_ref_list, axis=0)#(n, h, w)
        
        # example["instance_images_ref"] = torch.from_numpy(instance_images_ref).permute(0, 3, 1, 2)#(n, c, h, w)
        # example["mask_ref"] = torch.from_numpy(mask_ref)#(n, h, w)
        # example["instance_prompt_ref_ids"] = self.tokenizer(
        #     instance_prompt_ref_list,
        #     truncation=True,
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     return_tensors="pt",
        # ).input_ids#(n, 77)

        excluded_instance_images_dict = self.instance_images_dict[id_c].copy()
        excluded_instance_images_dict.pop(instance_image_path)
        instance_prompt_ref_samples = random.sample(list(excluded_instance_images_dict.items()), self.num_ref)

        def read_mask_and_image(instance_image_ref_path, instance_prompt_ref):
            instance_image_ref = Image.open(instance_image_ref_path)
            if not instance_image_ref.mode == "RGB":
                instance_image_ref = instance_image_ref.convert("RGB")

            # get ref mask
            (instance_name, instance_id) = str(instance_image_ref_path).split("/")[-2:]
            instance_ref_mask_path = osp.join(osp.dirname(str(instance_image_ref_path)), 'seg_masks', osp.splitext(instance_id)[0], instance_name.replace('_', ' ') + '.jpg')
            assert osp.exists(instance_ref_mask_path), f'{instance_ref_mask_path} not exists'
            instance_ref_mask = Image.open(instance_ref_mask_path)#(h,w)
            instance_ref_mask = np.array(instance_ref_mask).astype(np.uint8)#(h,w,3)
            instance_ref_mask = torch.Tensor(instance_ref_mask).permute(2,0,1)#(3,h,w)

            # apply resize augmentation and create a valid image region mask
            random_scale_ref = self.size
            if self.aug:
                random_scale_ref = (
                    np.random.randint(self.size // 3, self.size + 1)
                    if np.random.uniform() < 0.66
                    else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
                )
            instance_image_ref, instance_ref_mask, _ = self.preprocess_wi_mask(instance_image_ref, instance_ref_mask, random_scale_ref, self.interpolation)
            instance_ref_mask = F.interpolate(torch.Tensor(instance_ref_mask)[None, None, :, :], size=(self.mask_size, self.mask_size), mode='nearest')[0][0].numpy()

            if random_scale_ref < 0.6 * self.size:
                instance_prompt_ref = np.random.choice(["a far away ", "very small "]) + instance_prompt_ref
            elif random_scale_ref > self.size:
                instance_prompt_ref = np.random.choice(["zoomed in ", "close up "]) + instance_prompt_ref

            return instance_image_ref, instance_prompt_ref, instance_ref_mask
        
        # 4. read ref images in the same concept
        instance_image_ref_list, instance_prompt_ref_list, mask_ref_list = [], [], []
        for i, (instance_image_ref, instance_prompt_ref) in enumerate(instance_prompt_ref_samples):
            assert (not instance_image_ref == instance_image_path), f"{instance_image_ref} vs {instance_image_path}"
            instance_image_ref, instance_prompt_ref, instance_ref_mask = read_mask_and_image(instance_image_ref, instance_prompt_ref)
            instance_image_ref_list.append(instance_image_ref)
            instance_prompt_ref_list.append(instance_prompt_ref)
            mask_ref_list.append(instance_ref_mask)
        instance_images_ref = np.stack(instance_image_ref_list, axis=0)#(n, h, w, c)
        instance_ref_mask = np.stack(mask_ref_list, axis=0)#(n, h, w)
        
        example["instance_images_ref"] = torch.from_numpy(instance_images_ref).permute(0, 3, 1, 2)#(n, c, h, w)
        example["mask_ref"] = torch.from_numpy(instance_ref_mask)#(n, h, w)
        example["instance_prompt_ref_ids"] = self.tokenizer(
            instance_prompt_ref_list,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids#(n, 77)

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            class_images_ref_list, class_mask_ref_list, class_prompt_ref_list = [], [], []
            for _ in range(self.num_ref):
                class_images_ref_list.append(self.image_transforms_plus(class_image))
                class_mask_ref_list.append(torch.ones_like(example["mask"]))
                class_prompt_ref_list.append(class_prompt)
            class_images_ref = torch.stack(class_images_ref_list, axis=0)#(n, c, h, w)
            class_mask_ref = torch.stack(class_mask_ref_list, axis=0)#(n, h, w)

            example["class_images_ref"] = class_images_ref
            example["class_mask_ref"] = class_mask_ref
            example["class_prompt_ids_ref"] = self.tokenizer(
                class_prompt_ref_list,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids#(n, 77)

        return example

class ReferenceDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
        num_ref=1,
        placeholder_token="*",
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug
        self.num_ref = num_ref
        self.placeholder_token = placeholder_token

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        self.instance_images_dict = defaultdict(dict)
        for id_c, concept in enumerate(concepts_list):
            inst_img_path = [
                (id_c, x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            for x in Path(concept["instance_data_dir"]).iterdir():
                if x.is_file():
                    self.instance_images_dict[id_c].update({x: concept["instance_prompt"]})  
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)
        self.flip_p = 0.5 * hflip

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # augmentation for prior preservation
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        self.image_transforms_plus = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask
    
    def preprocess_wi_mask(self, image, mask, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        if isinstance(mask, torch.Tensor):
            mask = F.interpolate(mask.unsqueeze(0), (scale, scale), mode="nearest")[0][0]#(h,w)
            mask = mask.numpy().astype(np.uint8)

        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        instance_mask = np.zeros((self.size, self.size))
        image_mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            instance_mask = mask[top : top + inner, left : left + inner]
            image_mask = 255 * np.ones((self.size // factor, self.size // factor))
        elif scale < self.size:
            instance_image[top : top + inner, left : left + inner, :] = image
            instance_mask[top : top + inner, left : left + inner] = mask
            image_mask[
               top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 255.
        else:
            instance_image = image
            instance_mask = mask
            image_mask = image_mask + 255.
        return instance_image, instance_mask, image_mask

    def __getitem__(self, index):
        # 1. get instance image
        example = {}
        id_c, instance_image_path, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        #instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        #instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)
        
        # 2. get instance mask
        (instance_name, instance_id) = str(instance_image_path).split("/")[-2:]
        instance_mask_path = osp.join(osp.dirname(str(instance_image_path)), 'seg_masks', osp.splitext(instance_id)[0], instance_name.replace('_', ' ') + '.jpg')
        assert osp.exists(instance_mask_path), f'{instance_mask_path} not exists'
        instance_mask = Image.open(instance_mask_path)#(h,w)
        instance_mask = np.array(instance_mask).astype(np.uint8)#(h,w,3)
        instance_mask = torch.Tensor(instance_mask).permute(2,0,1)#(3,h,w)

        # if random.random() < self.flip_p:
        #     instance_image, instance_mask = TF.hflip(instance_image), TF.hflip(instance_mask)

        # disable aug
        instance_image, instance_mask, image_mask = self.preprocess_wi_mask(instance_image, instance_mask, self.size, self.interpolation)
        instance_mask = F.interpolate(torch.Tensor(instance_mask)[None, None, :, :], size=(self.mask_size, self.mask_size), mode='nearest')[0][0].numpy()
        #print(f'instance_mask:{instance_mask.shape}')

        #print(f'instance_prompt:{type(instance_prompt)}, {instance_prompt}')
        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["instance_mask"] = torch.from_numpy(instance_mask)#(h, w)
        example["mask"] = torch.from_numpy(image_mask)#(h, w)
        # example["instance_prompt_ids"] = self.tokenizer(
        #     instance_prompt,
        #     truncation=True,
        #     padding="max_length",
        #     max_length=self.tokenizer.model_max_length,
        #     return_tensors="pt",
        # ).input_ids
        example["instance_prompt_ids"], example["index"] = \
            self.obtain_text(instance_prompt)
        # print(instance_prompt, example["index"])

        excluded_instance_images_dict = self.instance_images_dict[id_c].copy()
        excluded_instance_images_dict.pop(instance_image_path)
        instance_prompt_ref_samples = random.sample(list(excluded_instance_images_dict.items()), self.num_ref)

        def read_mask_and_image(instance_image_ref_path, instance_prompt_ref):
            instance_image_ref = Image.open(instance_image_ref_path)
            if not instance_image_ref.mode == "RGB":
                instance_image_ref = instance_image_ref.convert("RGB")

            # get ref mask
            (instance_name, instance_id) = str(instance_image_ref_path).split("/")[-2:]
            instance_ref_mask_path = osp.join(osp.dirname(str(instance_image_ref_path)), 'seg_masks', osp.splitext(instance_id)[0], instance_name.replace('_', ' ') + '.jpg')
            assert osp.exists(instance_ref_mask_path), f'{instance_ref_mask_path} not exists'
            instance_ref_mask = Image.open(instance_ref_mask_path)#(h,w)
            instance_ref_mask = np.array(instance_ref_mask).astype(np.uint8)#(h,w,3)
            instance_ref_mask = torch.Tensor(instance_ref_mask).permute(2,0,1)#(3,h,w)

            # apply resize augmentation and create a valid image region mask
            random_scale_ref = self.size
            if self.aug:
                random_scale_ref = (
                    np.random.randint(self.size // 3, self.size + 1)
                    if np.random.uniform() < 0.66
                    else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
                )
            instance_image_ref, instance_ref_mask, _ = self.preprocess_wi_mask(instance_image_ref, instance_ref_mask, random_scale_ref, self.interpolation)
            instance_ref_mask = F.interpolate(torch.Tensor(instance_ref_mask)[None, None, :, :], size=(self.mask_size, self.mask_size), mode='nearest')[0][0].numpy()

            if random_scale_ref < 0.6 * self.size:
                instance_prompt_ref = np.random.choice(["a far away ", "very small "]) + instance_prompt_ref
            elif random_scale_ref > self.size:
                instance_prompt_ref = np.random.choice(["zoomed in ", "close up "]) + instance_prompt_ref

            return instance_image_ref, instance_prompt_ref, instance_ref_mask
        
        # 4. read ref images in the same concept
        instance_image_ref_list, instance_prompt_ref_list, mask_ref_list = [], [], []
        for i, (instance_image_ref, instance_prompt_ref) in enumerate(instance_prompt_ref_samples):
            assert (not instance_image_ref == instance_image_path), f"{instance_image_ref} vs {instance_image_path}"
            instance_image_ref, instance_prompt_ref, instance_ref_mask = read_mask_and_image(instance_image_ref, instance_prompt_ref)
            instance_image_ref_list.append(instance_image_ref)
            instance_prompt_ref_list.append(instance_prompt_ref)
            mask_ref_list.append(instance_ref_mask)
        instance_images_ref = np.stack(instance_image_ref_list, axis=0)#(n, h, w, c)
        instance_ref_mask = np.stack(mask_ref_list, axis=0)#(n, h, w)
        
        example["instance_images_ref"] = torch.from_numpy(instance_images_ref).permute(0, 3, 1, 2)#(n, c, h, w)
        example["mask_ref"] = torch.from_numpy(instance_ref_mask)#(n, h, w)
        example["instance_prompt_ref_ids"] = self.tokenizer(
            instance_prompt_ref_list,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids#(n, 77)
        

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"]) * 255.
            # example["class_prompt_ids"] = self.tokenizer(
            #     class_prompt,
            #     truncation=True,
            #     padding="max_length",
            #     max_length=self.tokenizer.model_max_length,
            #     return_tensors="pt",
            # ).input_ids
            example["class_prompt_ids"], example["class_index"] = \
                self.obtain_text(class_prompt)

            class_images_ref_list, class_mask_ref_list, class_prompt_ref_list = [], [], []
            for _ in range(self.num_ref):
                class_images_ref_list.append(self.image_transforms_plus(class_image))
                class_mask_ref_list.append(torch.ones_like(example["mask"])*255.)
                class_prompt_ref_list.append(class_prompt)
            class_images_ref = torch.stack(class_images_ref_list, axis=0)#(n, c, h, w)
            class_mask_ref = torch.stack(class_mask_ref_list, axis=0)#(n, h, w)

            example["class_images_ref"] = class_images_ref
            example["class_mask_ref"] = class_mask_ref
            example["class_prompt_ids_ref"] = self.tokenizer(
                class_prompt_ref_list,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids#(n, 77)

        return example

    def obtain_text(self, text, object_category=None):
        if object_category is None:
            placeholder_string = self.placeholder_token
        else:
            placeholder_string = object_category
        
        placeholder_index = 0
        words = text.strip().split(' ')
        # print(words, placeholder_string)
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx+1 #has start token
        
        index = torch.tensor(placeholder_index)

        input_ids = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids#(n, 77)
        return input_ids, index