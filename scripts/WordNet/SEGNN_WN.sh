DATA_DIR=dataset

MODEL_NAME=SEGNN
DATASET_NAME=WN18RR
DATA_PATH=$DATA_DIR/$DATASET_NAME
LITMODEL_NAME=SEGNNLitModel
TRAIN_SAMPLE_CLASS=SEGNNTrainSampler
TEST_SAMPLER_CLASS=SEGNNTestSampler
LOSS=Cross_Entropy_Loss
MAX_EPOCHS=800
EMB_DIM=200
TRAIN_BS=256
EVAL_BS=256
LR=1.5e-3
LABEL_SMOOTH=0.1
KG_LAYER=1
RM_RATE=0.5
ENT_DROP=0.2
REL_DROP=0
COMP_OP='mul'
K_H=10
K_W=20
ENT_DROP_PRED=0.3
FC_DROP=0.1
HID_DROP=0.4
INP_DROP=0
HID_SIZE=49000
KER_SZ=7
OUT_CHANNEL=250
CHECK_PER_EPOCH=10
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
    --bn \
    --pred_rel_w \
    --use_wandb \
    --save_config \
    --fc_bias \
    