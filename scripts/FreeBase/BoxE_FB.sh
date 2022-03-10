DATA_DIR=dataset

MODEL_NAME=BoxE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
MAX_EPOCHS=1000
EMB_DIM=1000
LOSS=Adv_Loss
ADV_TEMP=4.0
TRAIN_BS=1024
EVAL_BS=16
NUM_NEG=100
MARGIN=3.0
DIS_ORDER=2
LR=1e-4
REGULARIZATION=1e-7
CHECK_PER_EPOCH=50
NUM_WORKERS=16
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --dis_order $DIS_ORDER \
    --lr $LR \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    --save_config \

