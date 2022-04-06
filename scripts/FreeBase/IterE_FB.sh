DATA_DIR=dataset

MODEL_NAME=IterE
DATASET_NAME=FB15k-237-sparse
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=IterELitModel
MAX_EPOCHS=50
EMB_DIM=1024
LOSS=Adv_Loss
ADV_TEMP=1.0
TRAIN_BS=2048
EVAL_BS=16
NUM_NEG=128
MARGIN=0
LEARNING_RATE=1e-3
CHECK_PER_EPOCH=5
EARLY_STOP_PATIENCE=5
NUM_WORKERS=16
REGULARIZATION=0
GPU=2

MAX_ENTIALMENTS=1000
AXIOM_WEIGHT=0.1
AXIOM_PROBABILITY=0.9
UPDATE_AXIOM_PER=5


CUDA_VISIBLE_DEVICES=$GPU python -W ignore -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --loss_name $LOSS \
    --adv_temp $ADV_TEMP \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LEARNING_RATE \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --regularization $REGULARIZATION \
    --max_entialments $MAX_ENTIALMENTS \
    --axiom_weight $AXIOM_WEIGHT \
    --select_probability $AXIOM_PROBABILITY \
    --use_wandb \
    --update_axiom_per $UPDATE_AXIOM_PER \
    --use_weight \
    --save_config \
