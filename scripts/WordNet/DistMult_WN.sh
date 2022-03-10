DATA_DIR=dataset

MODEL_NAME=DistMult
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
MAX_EPOCHS=4000
EMB_DIM=1024
LOSS=Adv_Loss
TRAIN_BS=2048
EVAL_BS=16
NUM_NEG=50
MARGIN=200
LR=1e-4
REGULARIZATION=1e-5
CHECK_PER_EPOCH=50
NUM_WORKERS=16
GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --negative_adversarial_sampling \
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
