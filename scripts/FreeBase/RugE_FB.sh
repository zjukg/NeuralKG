#!/bin/sh
DATA_DIR=dataset

MODEL_NAME=RugE
DATASET_NAME=FB15K
TRAIN_BS=1024
NUM_NEG=10
DIM=200
MARGIN=12.0
ADV_TEMP=1.0
LEARNING_RATE=0.5
EPOCHES=1000
REGULARIZATION=0.0
MU=10
DATA_PATH=$DATA_DIR/$DATASET_NAME
SAVE_ID=1
GPU=1
SLACKNESS_PENALTY=0.01
LOSS_NAME=RugE_Loss
OPTIM_NAME=Adagrad
NUM_BATCHES=100
WEIGHT_DECAY=0.00000003
LITMODEL_NAME=RugELitModel
TRAIN_SAMPLER_CLASS=UniSampler

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --max_epochs $EPOCHES \
    --emb_dim $DIM \
    --num_neg $NUM_NEG \
    --train_bs $TRAIN_BS \
    --adv_temp $ADV_TEMP \
    --lr $LEARNING_RATE \
    -r $REGULARIZATION \
    --margin $MARGIN \
    --mu $MU \
    --slackness_penalty $SLACKNESS_PENALTY\
    --loss_name $LOSS_NAME\
    --optim_name $OPTIM_NAME\
    --num_batches $NUM_BATCHES\
    --weight_decay $WEIGHT_DECAY\
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --use_wandb \
    --save_config \

