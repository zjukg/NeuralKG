DATA_DIR=dataset

MODEL_NAME=BoxE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=1024
EVAL_BS=16
LOSS=Adv_Loss
NUM_NEG=100
DIM=1000
MARGIN=3.0
ADV_TEMP=4.0
LEARNING_RATE=0.00005
MAX_EPOCHS=1000
REGULARIZATION=1e-7
MU=10
NUM_WORKERS=8
CHECK_PER_EPOCH=50
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHS \
    --loss $LOSS \
    --emb_dim $DIM \
    --num_neg $NUM_NEG \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --adv_temp $ADV_TEMP \
    --lr $LEARNING_RATE \
    --margin $MARGIN \
    --num_workers $NUM_WORKERS \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --use_wandb

