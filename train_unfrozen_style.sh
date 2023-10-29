#!/bin/bash

# Set up a trap to gracefully exit upon receiving SIGINT (Ctrl+C)
trap "exit 0" SIGINT

while true; do
    # python -m ss_vq_vae.models.vqvae_oneshot --logdir=experiments/model-leaky-relu-unfrozen-style-encoder/ --pretrained_style_encoder=../COLA-PyTorch/run-25-07-2023/cola_306.pth --unfrozen_style_encoder --continue_training train
#     python -m ss_vq_vae.models.vqvae_oneshot --logdir=/mnt/vdb/model-leaky-relu-unfrozen-style-encoder-25-08-2023/ --pretrained_style_encoder=../COLA-PyTorch/run-25-07-2023/cola_306.pth --unfrozen_style_encoder --continue_training train
    # python -m ss_vq_vae.models.vqvae_oneshot --logdir=/mnt/vdb/model-leaky-relu-unfrozen-style-encoder-11-10-2023/ --pretrained_style_encoder=/mnt/vdb/run-30-08-2023/cola_306.pth --unfrozen_style_encoder --continue_training train
    python -m ss_vq_vae.models.vqvae_oneshot --logdir=/mnt/vdb/model-leaky-relu-no-style-pretraining-28-10-2023/ --unfrozen_style_encoder train

    # Optional: sleep for a short duration between iterations if needed
    # sleep 1
done
