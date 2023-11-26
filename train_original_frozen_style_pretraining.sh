#!/bin/bash

# Set up a trap to gracefully exit upon receiving SIGINT (Ctrl+C)
trap "exit 0" SIGINT
   
# python -m ss_vq_vae.models.vqvae_oneshot --model_type original --logdir=/mnt/vdb/model-original-frozen-style-pretraining-21-11-2023/ --pretrained_style_encoder=/mnt/vdb/run-contrastive-original-without-violin-bowed-21-11-2023/style_encoder_latest_6426.pth train

# Infinite loop
while true; do
    python -m ss_vq_vae.models.vqvae_oneshot --model_type original --logdir=/mnt/vdb/model-original-frozen-style-pretraining-21-11-2023/ --continue_training train
    # Optional: sleep for a short duration between iterations if needed
    echo "Time to sleep..."
    sleep 1
done
