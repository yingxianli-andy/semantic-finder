#!/bin/bash
# 运行实验1：SOTA性能对比
# 总共750个实验：1数据集 × 5方法 × 3预算 × 50seeds

cd /root/autodl-tmp/ijcai

# 启用网络加速（如果需要）
source /etc/network_turbo 2>/dev/null || true

# 运行实验1
python3 experiments/run_ijcai_experiments.py \
    --experiment 1 \
    --dataset ego-Twitter \
    --device cuda \
    --n_seeds 50 \
    --num_gpus 2 \
    --output_dir results/ijcai_experiments \
    --model_path results/models/final_model.pth

echo "实验1运行完成！"

