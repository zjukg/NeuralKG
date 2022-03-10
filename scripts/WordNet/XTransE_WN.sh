DATA_DIR=dataset

MODEL_NAME=XTransE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=4096
EVAL_BS=128
LITMODEL_NAME=XTransELitModel
TRAIN_SAMPLER_CLASS=XTransESampler
NUM_NEG=1
DIM=1000
LOSS_NAME=Margin_Loss
NUM_WORKERS=32
MARGIN=6.0
ADV_TEMP=0.5
LEARNING_RATE=0.001
MAX_EPOCHS=1000
REGULARIZATION=0
GPU='0'
CHECK_PER_EPOCH=10

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $DIM \
    --loss_name $LOSS_NAME \
    --num_neg $NUM_NEG \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --adv_temp $ADV_TEMP \
    --lr $LEARNING_RATE \
    --regularization $REGULARIZATION \
    --margin $MARGIN \
    --check_per_epoch $CHECK_PER_EPOCH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --num_workers $NUM_WORKERS \
    --use_wandb \

