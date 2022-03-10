DATA_DIR=dataset

MODEL_NAME=RGCN
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=RGCNLitModel
TRAIN_SAMPLER_CLASS=GraphSampler
TEST_SAMPLER_CLASS=GraphTestSampler
MAX_EPOCHES=10000
EMB_DIM=500
LOSS_NAME=RGCN_Loss
TRAIN_BS=60000
EVAL_BS=300
NUM_NEG=10
LR=1e-4
REGULARIZATION=1e-5
CHECK_PER_EPOCH=500
NUM_WORKERS=16
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -W ignore -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --max_epochs $MAX_EPOCHES \
    --emb_dim $EMB_DIM \
    --loss_name $LOSS_NAME \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --lr $LR \
    --regularization $REGULARIZATION \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    --save_config \

