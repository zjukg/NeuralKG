DATA_DIR=dataset
DATASET_NAME=WN18RR_v1
MODEL_NAME=MorsE

DATA_PATH=${DATA_DIR}/${DATASET_NAME}
DB_PATH=${DATA_DIR}/${DATASET_NAME}_meta_subgraph
PK_PATH=${DATA_DIR}/${DATASET_NAME}.pkl
EVAL_TASK=link_prediction
TRAIN_SAMPLER_CLASS=MetaSampler
VALID_SAMPLER_CLASS=ValidMetaSampler
TEST_SAMPLER_CLASS=TestMetaSampler
LITMODEL_NAME=MetaGNNLitModel
LOSS=Adv_Loss
NUM_LAYERS=3
NUM_BASES=4
MAX_EPOCHS=10
EMB_DIM=32
TRAIN_BS=64
EVAL_BS=64
TEST_BS=512
NUM_NEG=32
MARGIN=10.0
LR=1e-2
CHECK_PER_STEP=5
EARLY_STOP_PATIENCE=1000
NUM_WORKERS=20
DROPOUT=0
CALC_HITS=1,5,10
KGE_MODEL=RotatE
GPU=1

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --db_path $DB_PATH \
    --pk_path $PK_PATH \
    --eval_task $EVAL_TASK \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --valid_sampler_class $VALID_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --litmodel_name $LITMODEL_NAME \
    --loss $LOSS \
    --num_layers $NUM_LAYERS \
    --num_bases $NUM_BASES \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --test_bs $TEST_BS \
    --num_neg $NUM_NEG \
    --margin $MARGIN \
    --lr $LR \
    --check_per_step $CHECK_PER_STEP \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --num_workers $NUM_WORKERS \
    --dropout $DROPOUT \
    --calc_hits $CALC_HITS \
    --kge_model $KGE_MODEL \
    --inductive \

