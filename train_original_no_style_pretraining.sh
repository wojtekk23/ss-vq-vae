#!/bin/bash

# Set up a trap to gracefully exit upon receiving SIGINT (Ctrl+C)
trap "exit 0" SIGINT

# Infinite loop
# while true; do
    python -m ss_vq_vae.models.vqvae_oneshot --model_type original --logdir=/mnt/vdb/model-original-no-style-pretraining-19-11-2023/ --unfrozen_style_encoder --continue_training train
    # Optional: sleep for a short duration between iterations if needed
    echo "Time to sleep..."
    sleep 10
# done
