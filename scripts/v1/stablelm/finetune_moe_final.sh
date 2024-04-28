#!/bin/bash

# --deepspeed ./scripts/zero2.json \

# Run under cloned MoE-LLaVA repo
moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
JSON_FOLDER="json_folder/okvqa_train_jsons"
IMAGE_FOLDER="image_folder"

# GP 1

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 deepspeed --master_port 29600 moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --model_name_or_path /home/ubuntu/workspace/models/MoE-LLaVA-StableLM-1.6B-4e \
    --version stablelm \
    --data_path ${JSON_FOLDER}/gp1_train_10.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output_trained/MoE-LLaVA-StableLM-1.6B-4e-okvqa_gp1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

# GP 2

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 deepspeed --master_port 29600 moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --model_name_or_path /home/ubuntu/workspace/models/MoE-LLaVA-StableLM-1.6B-4e \
    --version stablelm \
    --data_path ${JSON_FOLDER}/gp2_train_10.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output_trained/MoE-LLaVA-StableLM-1.6B-4e-okvqa_gp2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

# GP 3

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 deepspeed --master_port 29600 moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --model_name_or_path /home/ubuntu/workspace/models/MoE-LLaVA-StableLM-1.6B-4e \
    --version stablelm \
    --data_path ${JSON_FOLDER}/gp3_train_10.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output_trained/MoE-LLaVA-StableLM-1.6B-4e-okvqa_gp3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"

cp /home/ubuntu/workspace/MoE-llava-model-files/* ./output_trained/MoE-LLaVA-StableLM-1.6B-4e-okvqa_gp1

cp /home/ubuntu/workspace/MoE-llava-model-files/* ./output_trained/MoE-LLaVA-StableLM-1.6B-4e-okvqa_gp2

cp /home/ubuntu/workspace/MoE-llava-model-files/* ./output_trained/MoE-LLaVA-StableLM-1.6B-4e-okvqa_gp3

/home/ubuntu/miniconda3/envs/moellava/bin/python /home/ubuntu/workspace/MoE-LLaVA/eval_script.py
