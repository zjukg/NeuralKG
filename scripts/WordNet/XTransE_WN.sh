DATA_DIR=dataset

MODEL_NAME=XTransE
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=XTransELitModel
TRAIN_SAMPLER_CLASS=XTransESampler
LOSS_NAME=Margin_Loss
ADV_TEMP=0.5
MAX_EPOCHS=1000
EMB_DIM=500
TRAIN_BS=4096
EVAL_BS=128
NUM_NEG=1
MARGIN=6.0
LEARNING_RATE=0.001
REGULARIZATION=0
CHECK_PER_EPOCH=10
NUM_WORKERS=32
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --loss_name $LOSS_NAME \
    --adv_temp $ADV_TEMP \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LEARNING_RATE \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    --save_config \

