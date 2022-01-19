DATA_DIR=dataset

MODEL_NAME=CrossE
LITMODEL_NAME=CrossELitModel
TRAIN_SAMPLER_CLASS=CrossESampler
LOSS=CrossE_Loss
DATASET_NAME=WN18
DATA_PATH=$DATA_DIR/$DATASET_NAME

EVAL_BS=512
TRAIN_BS=2048
DIM=100
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-4

MAX_EPOCHES=1000
CHECK_PER_EPOCH=5
EARLY_STOP=5000
NUM_WPRDKERS=8
GPU=2


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --loss $LOSS \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHES \
    --litmodel_name $LITMODEL_NAME \
    --emb_dim $DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --lr $LEARNING_RATE \
    --check_per_epoch $CHECK_PER_EPOCH \
    --weight_decay $WEIGHT_DECAY \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --use_wandb \
    --early_stop_patience $EARLY_STOP \
    





