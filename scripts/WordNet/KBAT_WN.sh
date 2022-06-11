DATA_DIR=dataset

MODEL_NAME=KBAT
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KBATLitModel
TRAIN_SAMPLER_CLASS=KBATSampler
TEST_SAMPLER_CLASS=TestSampler
LOSS_NAME=KBAT_Loss
MAX_EPOCHS=6000
EPOCH_GAT=800
EMB_DIM=100
TRAIN_BS=2048
EVAL_BS=128
NUM_NEG=20
MARGIN=5.0
LEARNING_RATE=0.001
NUM_WORKERS=40
CHECK_PER_EPOCH=100
EARLY_STOP_PATIENCE=10
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --loss_name $LOSS_NAME \
    --max_epochs $MAX_EPOCHS \
    --epoch_GAT $EPOCH_GAT \
    --emb_dim $EMB_DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --check_per_epoch $CHECK_PER_EPOCH \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --use_wandb \
    --save_config \
    



