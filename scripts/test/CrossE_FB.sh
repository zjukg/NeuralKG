DATA_DIR=dataset

MODEL_NAME=CrossE
LITMODEL_NAME=CrossELitModel
DATASET_NAME=FB15K237
TRAIN_SAMPLER_CLASS=CrossESampler
LOSS=CrossE_Loss
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=4000
EVAL_BS=512
DIM=300
LEARNING_RATE=0.01
WEIGHT_DECAY=1e-6
MAX_EPOCHES=1000
CHECK_PER_EPOCH=1
EARLY_STOP=5000
NUM_WPRDKERS=16
HITS=1,3,10
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
    --early_stop_patience $EARLY_STOP \
    --calc_hits $HITS \
    --use_wandb \
    # --num_workers $NUM_WPRDKERS \
    # --save_config \
    # --use_wandb \





