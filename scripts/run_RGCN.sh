DATA_DIR=dataset

MODEL_NAME=RGCN
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LOSS_NAME=RGCN_Loss
TRAIN_BS=60000
EVAL_BS=300
DIM=500
LEARNING_RATE=1e-4
MAX_EPOCHES=10000
REGULARIZATION=1e-4
NUM_WORKERS=8
NUM_NEG=10
CHECK_PER_EPOCH=500
LITMODEL_NAME=RGCNLitModel
TRAIN_SAMPLER_CLASS=GraphSampler
TEST_SAMPLER_CLASS=GraphTestSampler
GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --loss_name $LOSS_NAME \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --emb_dim $DIM \
    --lr $LEARNING_RATE \
    --max_epochs $MAX_EPOCHES \
    --regularization $REGULARIZATION \
    --num_workers $NUM_WORKERS \
    --num_neg $NUM_NEG \
    --check_per_epoch $CHECK_PER_EPOCH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --use_wandb \


