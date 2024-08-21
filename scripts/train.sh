#!/bin/bash

export PROJECT_ROOT="/maliyuan/video_projects/EqDiff"
export CKP_ROOT="/qiguojun/home/Models"

cd  $PROJECT_ROOT

# DATASET_NAME=${1:-"dog2"}
# PROMPT_NAME=${2:-"dog"}
# INIT_TOKEN=${3:-"dog"}
DATASET_NAME=${1:-"cat_toy"}
PROMPT_NAME=${2:-"cat toy"}
INIT_TOKEN=${3:-"cat_toy"}
guidance_scale=${4:-12.5}
phase1_train_steps=${5:-0}
phase2_train_steps=${6:-0}
phase3_train_steps=${7:-0}
phase4_train_steps=${8:-600}
dilate_iters=${9:-5}
LF=${10:-0.1}

HF=${11:-0.1}
AllF=${12:-0.8}

export SD14_PATH="${CKP_ROOT}/CompVis/stable-diffusion-v1-4"
export SD15_PATH="${CKP_ROOT}/runwayml/stable-diffusion-v1-5"
export CLIP_PATH="${CKP_ROOT}/openai/clip-vit-large-patch14"
export PATH_PREFIX="p1step$phase1_train_steps-p2step$phase2_train_steps-p3step$phase3_train_steps-p4step$phase4_train_steps-dilate_iters_$dilate_iters-Freq_$LF-$HF-$AllF-_"
export OUTPUT_DIR="./checkpoints/${PATH_PREFIX}$DATASET_NAME"
export INSTANCE_DIR="$PROJECT_ROOT/data/$DATASET_NAME"
export PROMPTS_FILE="$PROJECT_ROOT/prompts/base_prompt.txt"

accelerate launch ./train.py \
  --pretrained_model_name_or_path=$SD14_PATH  \
  --pretrained_model_name_or_path_for_img_gen=$SD15_PATH \
  --reference_encoder_path=$SD14_PATH \
  --pretrained_clip_path=$CLIP_PATH \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir="$PROJECT_ROOT/real_reg/sample_$DATASET_NAME" \
  --class_prompt=$INIT_TOKEN --num_class_images=200 \
  --instance_prompt="photo of a <new2> $PROMPT_NAME"  \
  --resolution=512  \
  --train_batch_size=2  \
  --initial_learning_rate=1e-5 \
  --learning_rate=1e-5  \
  --phase3_learning_rate=1e-5  \
  --phase4_learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --scale_lr \
  --modifier_token "<new2>" \
  --validation_prompt="A photo of <new2> $INIT_TOKEN in the beach" \
  --num_validation_images=2 \
  --report_to="tensorboard" \
  --no_safe_serialization \
  --config "$PROJECT_ROOT/configs/reference_diffusion_$DATASET_NAME.yaml" \
  --checkpointing_steps 400 \
  --validation_steps 100 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --align_loss_weight=1.0 \
  --phase1_train_steps=$phase1_train_steps \
  --phase2_train_steps=$phase2_train_steps \
  --phase3_train_steps=$phase3_train_steps \
  --phase4_train_steps=$phase4_train_steps \
  --lambda_attention=1e-1 \
  --lambda_sattention=1e-1 \
  --enable_phase3_training \
  --training_log_steps=50 \
  --high_freq_percentage=5 \
  --low_freq_percentage=50 \
  --gray_rate=0.5 \
  --grayprob=1.0 \
  --text_inj \
  --time_gap=2 \
  --use_gray \
  --mask_noise \
  --dilate_iters=$dilate_iters \
  --LF=$LF \
  --HF=$HF \
  --AllF=$AllF \
  --use_reference_encoder \