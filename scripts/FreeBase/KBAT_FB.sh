DATA_DIR=dataset

DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
MODEL_NAME=KBAT
LOSS_NAME=KBAT_Loss
LITMODEL_NAME=KBATLitModel
TRAIN_SAMPLER_CLASS=KBATSampler
TEST_SAMPLER_CLASS=TestSampler
TRAIN_BS=2048
NUM_WORKERS=16
EVAL_BS=128
DIM=100
MAX_EPOCHS=3400
NUM_NEG=20
LEARNING_RATE=0.001
MARGIN=1.0
CHECK_PER_EPOCH=3400
GPU=3

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
    --lr $LEARNING_RATE \
    --margin $MARGIN \
    --check_per_epoch $CHECK_PER_EPOCH \
    --partial_2hop \
    --use_wandb \
    --save_config \
    



