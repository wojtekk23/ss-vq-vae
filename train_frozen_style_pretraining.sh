#!/bin/bash

# Set up a trap to gracefully exit upon receiving SIGINT (Ctrl+C)
trap "exit 0" SIGINT

# Infinite loop
# while true; do
    # Replace "your_command_here" with the command you want to execute
    # python -m ss_vq_vae.models.vqvae_oneshot --logdir=experiments/model-leaky-relu/ --pretrained_style_encoder=../COLA-PyTorch/run-25-07-2023/cola_306.pth --continue_training train
    # python -m ss_vq_vae.models.vqvae_oneshot --logdir=/mnt/vdb/model-leaky-relu-30-08-2023/ --pretrained_style_encoder=/mnt/vdb/run-30-08-2023/cola_306.pth --continue_training train
    # python -m ss_vq_vae.models.vqvae_oneshot --logdir=/mnt/vdb/model-leaky-relu-17-10-2023/ --pretrained_style_encoder=/mnt/vdb/run-30-08-2023/cola_306.pth --continue_training train
    COLA_VISIBLE_DEVICES=0 python -m ss_vq_vae.models.vqvae_oneshot --logdir=/mnt/vdc/model-leaky-relu-frozen-style-pretraining-01-09-2024/ --pretrained_style_encoder=/mnt/vdb/run-without-violin-bowed-27-10-2023/cola_305.pth train
    # Optional: sleep for a short duration between iterations if needed
    echo "Time to sleep..."
    sleep 2
# done
