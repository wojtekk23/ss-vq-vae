# MODEL_NAME=model-leaky-relu-finetuned-style-pretraining-29-08-2024
LOGDIR_PATH=$1
MODEL_TYPE=$2
MODEL_NAME=$(basename $LOGDIR_PATH)

time python -m ss_vq_vae.models.vqvae_oneshot --logdir=$LOGDIR_PATH --model_type=$MODEL_TYPE run /mnt/vdb/test_set_pairs.csv outputs/$MODEL_NAME/vqvae_list outputs/$MODEL_NAME/vqvae
