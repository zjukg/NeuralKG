DATA_DIR=dataset

MODEL_NAME=RMPI
DATASET_NAME=WN18RR_v2
DATA_PATH=${DATA_DIR}/${DATASET_NAME}
DB_PATH=${DATA_DIR}/${DATASET_NAME}_RMPI_subgraph
PK_PATH=$DATA_DIR/${DATASET_NAME}.pkl
TRAIN_SAMPLER_CLASS=RMPISampler
VALID_SAMPLER_CLASS=ValidRMPISampler
TEST_SAMPLER_CLASS=TestRMPISampler
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
CHECK_PER_STEP=455
EARLY_STOP_PATIENCE=20
NUM_WORKERS=20
CALC_HITS=1,5,10
GPU=1
NUM_LAYERS=3
NUM_BASES=4
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
    --check_per_step $CHECK_PER_STEP \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --num_workers $NUM_WORKERS \
    --calc_hits $CALC_HITS \
    --num_layers $NUM_LAYERS \
    --num_bases $NUM_BASES \
    --hop $HOP \
    --enclosing_sub_graph $ENCLOSING_SUB_GRAPH \
    --ablation $ABLATION \
    --l2 $L2 \
    --inductive \
    --conc \


