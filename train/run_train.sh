#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用的GPU编号
export MASTER_PORT=29500  # master端口
export MASTER_ADDR=localhost  # master地址
export WORLD_SIZE=1  # 总进程数（GPU数量）
export NODE_RANK=0  # 当前节点的rank
export LOCAL_RANK=0  # 本地rank

# 创建必要的目录
mkdir -p logs
mkdir -p results
mkdir -p models/Qwen-finetuned

# 启动分布式训练
deepspeed --num_gpus=1 \
    --master_port=$MASTER_PORT \
    finetune_deepspeed.py \
    --deepspeed ds_config.json