DATA_DIR=dataset

MODEL_NAME=ComplEx_NNE_AER
DATASET_NAME=FB15K
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
MAX_EPOCHS=1000
EMB_DIM=1000
LOSS=ComplEx_NNE_AER_Loss
ADV_TEMP=4.0
TRAIN_BS=1024
EVAL_BS=16
NUM_NEG=100
MARGIN=3.0
LR=1e-4
REGULARIZATION=1e-7
CHECK_PER_EPOCH=50
NUM_WORKERS=16
GPU=0

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
    --lr $LR \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    --save_config \

