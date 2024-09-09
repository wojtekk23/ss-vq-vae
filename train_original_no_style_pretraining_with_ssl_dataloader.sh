#!/bin/bash

# Set up a trap to gracefully exit upon receiving SIGINT (Ctrl+C)
trap "exit 0" SIGINT

# Infinite loop
# while true; do
    python -m ss_vq_vae.models.vqvae_oneshot --model_type original --logdir=/mnt/vdb/model-original-no-style-pretraining-with-ssl-dataloader-07-09-2024/ --unfrozen_style_encoder --dataset_type ssl train
    # Optional: sleep for a short duration between iterations if needed
    echo "Time to sleep..."
    sleep 10
# done
