DATA_DIR=dataset

DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
MODEL_NAME=CompGCN
LOSS_NAME=Cross_Entropy_Loss
LITMODEL_NAME=CompGCNLitModel
TRAIN_SAMPLER_CLASS=CompGCNSampler
TEST_SAMPLER_CLASS=CompGCNTestSampler
TRAIN_BS=2048
NUM_WORKERS=16
EVAL_BS=256
DIM=100
MAX_EPOCHS=2000
NUM_NEG=1
CHECK_PER_EPOCH=50
LEARNING_RATE=0.0001
DECODER_MODEL=ConvE
OPN=corr
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --loss_name $LOSS_NAME \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --train_bs $TRAIN_BS \
    --num_workers $NUM_WORKERS \
    --eval_bs $EVAL_BS \
    --emb_dim $DIM \
    --max_epochs $MAX_EPOCHS \
    --num_neg $NUM_NEG \
    --check_per_epoch $CHECK_PER_EPOCH \
    --lr $LEARNING_RATE \
    --decoder_model $DECODER_MODEL \
    --opn $OPN \
    --use_wandb \



