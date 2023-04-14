DATA_DIR=dataset

MODEL_NAME=SEGNN
DATASET_NAME=FB15K237
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=SEGNNLitModel
TRAIN_SAMPLE_CLASS=SEGNNTrainSampler
TEST_SAMPLER_CLASS=SEGNNTestSampler
LOSS=Cross_Entropy_Loss
MAX_EPOCHS=600
EMB_DIM=450
TRAIN_BS=1024
EVAL_BS=1024
LR=3.5e-4
LABEL_SMOOTH=0.1
KG_LAYER=2
RM_RATE=0.5
ENT_DROP=0.3
REL_DROP=0.1
COMP_OP='mul'
K_H=15
K_W=30
ENT_DROP_PRED=0.3
FC_DROP=0.3
HID_DROP=0.5
INP_DROP=0
HID_SIZE=105800
KER_SZ=8
OUT_CHANNEL=200
CHECK_PER_EPOCH=1
NUM_WORKERS=10
GPU=0


CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --litmodel_name $LITMODEL_NAME \
    --train_sampler_class $TRAIN_SAMPLE_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --loss_name $LOSS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --lr $LR \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --label_smooth $LABEL_SMOOTH \
    --kg_layer $KG_LAYER \
    --rm_rate $RM_RATE \
    --inp_drop $INP_DROP \
    --ent_drop $ENT_DROP \
    --rel_drop $REL_DROP \
    --comp_op $COMP_OP \
    --hid_size $HID_SIZE \
    --k_h $K_H \
    --k_w $K_W \
    --ent_drop_pred $ENT_DROP_PRED \
    --fc_drop $FC_DROP \
    --hid_drop $HID_DROP \
    --ker_sz $KER_SZ \
    --out_channel $OUT_CHANNEL \
    --check_per_epoch $CHECK_PER_EPOCH \
    --num_workers $NUM_WORKERS \
    --use_wandb \
    --save_config \
    --fc_bias \
    