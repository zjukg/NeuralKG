DATA_DIR=dataset

MODEL_NAME=RMPI
DATASET_NAME=WN18RR_v4
DATA_PATH=${DATA_DIR}/${DATASET_NAME}
DB_PATH=${DATA_DIR}/${DATASET_NAME}_RMPI_subgraph
PK_PATH=${DATA_DIR}/${DATASET_NAME}.pkl
TEST_DB_PATH=${DATA_DIR}/${DATASET_NAME}_test_subgraph
TRAIN_SAMPLER_CLASS=RMPISampler
VALID_SAMPLER_CLASS=ValidRMPISampler
TEST_SAMPLER_CLASS=TestRMPISampler_hit
LITMODEL_NAME=indGNNLitModel
EVAL_TASK=link_prediction
LOSS=Margin_Loss
MAX_EPOCHS=20
EMB_DIM=32
TRAIN_BS=16
EVAL_BS=16
TEST_BS=1
MARGIN=10.0
LR=1e-3
REDUCTION=sum
CHECK_PER_STEP=455
EARLY_STOP_PATIENCE=20
NUM_WORKERS=20
CALC_HITS=1,5,10
GPU=0
HOP=2
ENCLOSING_SUB_GRAPH=False
ABLATION=1
L2=5e-2

CUDA_VISIBLE_DEVICES=$GPU python -u main.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --data_path $DATA_PATH \
    --db_path $DB_PATH \
    --pk_path $PK_PATH \
    --test_db_path $TEST_DB_PATH \
    --eval_task $EVAL_TASK \
    --train_sampler_class $TRAIN_SAMPLER_CLASS \
    --valid_sampler_class $VALID_SAMPLER_CLASS \
    --test_sampler_class $TEST_SAMPLER_CLASS \
    --litmodel_name $LITMODEL_NAME \
    --loss $LOSS \
    --max_epochs $MAX_EPOCHS \
    --emb_dim $EMB_DIM \
    --train_bs $TRAIN_BS \
    --eval_bs $EVAL_BS \
    --test_bs $TEST_BS \
    --margin $MARGIN \
    --lr $LR \
    --reduction $REDUCTION \
    --check_per_step $CHECK_PER_STEP \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --num_workers $NUM_WORKERS \
    --calc_hits $CALC_HITS \
    --hop $HOP \
    --enclosing_sub_graph $ENCLOSING_SUB_GRAPH \
    --ablation $ABLATION \
    --l2 $L2 \
    --target2nei_atten \
    --inductive \
    --conc \
    



