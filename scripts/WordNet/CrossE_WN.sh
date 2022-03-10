DATA_DIR=dataset

MODEL_NAME=CrossE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=CrossELitModel
TRAIN_SAMPLER_CLASS=CrossESampler
MAX_EPOCHES=1000
EMB_DIM=100
LOSS=CrossE_Loss
TRAIN_BS=2048
EVAL_BS=512
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-4
CHECK_PER_EPOCH=5
EARLY_STOP=5000
NUM_WPRDKERS=8
GPU=2


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --max_epochs $MAX_EPOCHES \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --check_per_epoch $CHECK_PER_EPOCH \
    --early_stop_patience $EARLY_STOP \
    --num_workers $NUM_WPRDKERS \
    --use_wandb \
    --save_config \
    





