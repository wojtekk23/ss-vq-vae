LOGDIR_PATH=$1
MODEL_TYPE=$2
MODEL_NAME=$(basename $LOGDIR_PATH)

time python -m ss_vq_vae.models.vqvae_oneshot --logdir=$LOGDIR_PATH --model_type=$MODEL_TYPE run /mnt/vdb/validation_set_2_pairs.csv outputs/$MODEL_NAME/val2/vqvae_list outputs/$MODEL_NAME/val2/vqvae
