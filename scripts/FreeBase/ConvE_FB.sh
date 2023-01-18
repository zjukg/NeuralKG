DATA_DIR=dataset

MODEL_NAME=ConvE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LOSS=Cross_Entropy_Loss
TRAIN_BS=1024
EVAL_BS=256
DIM=200
LEARNING_RATE=0.001
MAX_EPOCHES=3000
REGULARIZATION=0
NUM_WORKERS=8
CHECK_PER_EPOCH=30
LITMODEL_NAME=ConvELitModel
TRAIN_SAMPLER_CLASS=ConvSampler
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --loss $LOSS \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --emb_dim $DIM \
    --lr $LEARNING_RATE \
    --max_epochs $MAX_EPOCHES \
    --regularization $REGULARIZATION \
    --num_workers $NUM_WORKERS \
    --check_per_epoch $CHECK_PER_EPOCH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --use_wandb \
    --save_config





