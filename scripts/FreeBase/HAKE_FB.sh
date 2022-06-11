DATA_DIR=dataset

MODEL_NAME=HAKE
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=KGELitModel
MAX_EPOCHS=1000
EMB_DIM=1000
LOSS=Adv_Loss
ADV_TEMP=1.0
TRAIN_BS=1024
EVAL_BS=16
NUM_NEG=256
MARGIN=9.0
LR=5e-5
CHECK_PER_EPOCH=30
PHASE_WEIGHT=1.0
MODULUS_WEIGHT=3.5
NUM_WORKERS=40
GPU=2


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss $LOSS \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --check_per_epoch $CHECK_PER_EPOCH \
    --phase_weight $PHASE_WEIGHT \
    --modulus_weight $MODULUS_WEIGHT \
    --num_workers $NUM_WORKERS \
    --use_weight \
    --use_wandb \
    --save_config \
