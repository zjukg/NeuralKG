DATA_DIR=dataset

MODEL_NAME=IterE
LITMODEL_NAME=IterELitModel
LOSS=IterE_Loss
DATASET_NAME=WN18RR-sparse
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=1024
NUM_NEG=256
DIM=200
MARGIN=1
ADV_TEMP=1.0
LEARNING_RATE=1e-3
MAX_EPOCHS=380
EARLY_STOP_PATIENCE=200
REGULARIZATION=1e-5
EARLY_STOP=100
SEED=11
GPU=1
CHECK_PER_EPOCH=1

MAX_ENTIALMENTS=10000
AXIOM_WEIGHT=0.1
AXIOM_PROBABILITY=0.9


CUDA_VISIBLE_DEVICES=$GPU python -W ignore -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --loss_name $LOSS \
    --data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHS \
    --litmodel_name $LITMODEL_NAME \
    --emb_dim $DIM \
    --num_neg $NUM_NEG \
    --train_bs $TRAIN_BS \
    --adv_temp $ADV_TEMP \
    --lr $LEARNING_RATE \
    --regularization $REGULARIZATION \
    --margin $MARGIN \
    --check_per_epoch $CHECK_PER_EPOCH \
    --axiom_weight $AXIOM_WEIGHT \
    --select_probability $AXIOM_PROBABILITY \
    --max_entialments $MAX_ENTIALMENTS \
    --early_stop_patience $EARLY_STOP_PATIENCE

