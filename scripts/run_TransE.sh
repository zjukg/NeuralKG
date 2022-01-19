DATA_DIR=dataset

MODEL_NAME=TransE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
TRAIN_BS=1024
NUM_NEG=256
DIM=1000
MARGIN=9.0
LEARNING_RATE=1e-4
MAX_EPOCHES=100
EVAL_BS=16
CHECK_PER_EPOCH=5
EARLY_STOP=32
ADV_TEMP=1.5
CACHE_SIZE=50
UPDATE_CACHE_EPOCH=5
ALPHA=1
WARMUP=15
GPU=3


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHES \
    --emb_dim $DIM \
    --litmodel_name $LITMODEL_NAME \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --lr $LEARNING_RATE \
    --check_per_epoch $CHECK_PER_EPOCH \
    --early_stop_patience $EARLY_STOP \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --use_wandb \
    --adv_temp $ADV_TEMP \
    #--wandb_offline \
    #--calc_filter \
    #--cache_size $CACHE_SIZE \
    #--update_cache_epoch $UPDATE_CACHE_EPOCH \
    #--alpha $ALPHA \
    #--warmup $WARMUP \
    # --test_only \
    # --limit_train_batches 2 \
    # --leakage \
    # --save_config \





