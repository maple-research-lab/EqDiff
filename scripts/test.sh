#!/bin/bash
if [ -z "$PROJECT_ROOT" ]; then
    export PROJECT_ROOT="/fangxueji/projects_archived/EqDiff"
fi
cd  $PROJECT_ROOT

DATASET_NAME=${1:-"can"}
PROMPT_NAME=${2:-"can"}
INIT_TOKEN=${3:-"can"}
guidance_scale=${4:-7.5}
phase1_train_steps=${5:-0}
phase2_train_steps=${6:-0}
phase3_train_steps=${7:-0}
phase4_train_steps=${8:-600}
dilate_iters=${9:-5}
LF=${10:-0.1}
HF=${11:-0.1}
AllF=${12:-0.8}

export PATH_PREFIX="p1step$phase1_train_steps-p2step$phase2_train_steps-p3step$phase3_train_steps-p4step$phase4_train_steps-dilate_iters_$dilate_iters-Freq_$LF-$HF-$AllF-_"
export SD14_PATH="/qiguojun/home/Models/CompVis/stable-diffusion-v1-4"

# style mode
accelerate launch test.py \
    --DATASET_NAME=$DATASET_NAME \
    --PROMPT_NAME="$PROMPT_NAME" \
    --PATH_PREFIX=$PATH_PREFIX \
    --MODE="style" \
    --SD_PATH=$SD14_PATH \
    --config="configs/reference_diffusion_$DATASET_NAME.yaml" \
    --modifier_token "<new2>" \
    --guidance_scale=$guidance_scale \
    --use_reference_encoder \

# no style mode
accelerate launch test.py \
    --DATASET_NAME=$DATASET_NAME \
    --PROMPT_NAME="$PROMPT_NAME" \
    --PATH_PREFIX=$PATH_PREFIX \
    --MODE="None" \
    --SD_PATH=$SD14_PATH \
    --config="configs/reference_diffusion_$DATASET_NAME.yaml" \
    --modifier_token "<new2>" \
    --guidance_scale=$guidance_scale \
    --use_reference_encoder \