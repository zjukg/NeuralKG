DATA_DIR=dataset

MODEL_NAME=TransE
LOSS=Adv_Loss
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
TRAIN_BS=1024
NUM_NEG=128
DIM=400
MARGIN=9.0
ADV_TEMP=1.0
LEARNING_RATE=1e-4
MAX_EPOCHS=380
# MAX_EPOCHS=480
CHECK_PER_EPOCH=5
REGULARIZATION=1e-5
EARLY_STOP=100
GPU=2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $DIM \
    --num_neg $NUM_NEG \
    --train_bs $TRAIN_BS \
    --adv_temp $ADV_TEMP \
    --lr $LEARNING_RATE \
    -r $REGULARIZATION \
    --margin $MARGIN \
    --check_per_epoch $CHECK_PER_EPOCH \
    --early_stop_patience $EARLY_STOP \
    --use_weight
    --limit_train_batches 10 \
    # --seed $SEED \
    # --loss_name $LOSS \
    # --limit_val_batches 10 \
