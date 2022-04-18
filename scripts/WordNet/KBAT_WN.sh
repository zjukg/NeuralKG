DATA_DIR=dataset

MODEL_NAME=KBAT
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KBATLitModel
TRAIN_SAMPLER_CLASS=KBATSampler
TEST_SAMPLER_CLASS=TestSampler
LOSS_NAME=KBAT_Loss
MAX_EPOCHS=3400
EMB_DIM=100
TRAIN_BS=2048
EVAL_BS=128
NUM_NEG=20
MARGIN=1.0
LEARNING_RATE=0.001
NUM_WORKERS=16
CHECK_PER_EPOCH=3400
GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --loss_name $LOSS_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --limit_val_batches $LIMIT_VAL_BATCHES \
    --check_per_epoch $CHECK_PER_EPOCH \
    --use_wandb \
    --save_config \
    



