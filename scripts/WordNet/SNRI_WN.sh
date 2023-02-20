DATA_DIR=dataset
DATASET_NAME=WN18RR_v2
MODEL_NAME=SNRI

DATA_PATH=${DATA_DIR}/${DATASET_NAME}
DB_PATH=${DATA_DIR}/${DATASET_NAME}_subgraph
PK_PATH=${DATA_DIR}/${DATASET_NAME}.pkl
TEST_DB_PATH=${DATA_DIR}/${DATASET_NAME}_triple
EVAL_TASK=link_prediction
TRAIN_SAMPLER_CLASS=SubSampler
VALID_SAMPLER_CLASS=ValidSampler
TEST_SAMPLER_CLASS=TestSampler_hit
LITMODEL_NAME=SNRILitModel
LOSS=Margin_Loss
NUM_LAYERS=3
NUM_BASES=4
MAX_EPOCHS=50
EMB_DIM=32
TRAIN_BS=16
EVAL_BS=16
TEST_BS=1
NUM_NEG=100
MARGIN=10.0
LR=1e-3
CHECK_PER_STEP=300
EARLY_STOP_PATIENCE=20
NUM_WORKERS=20
DROPOUT=0
CALC_HITS=1,5,10
GPU=1
CHECKPOINT_DIR=/home/lli/NeuralKG/output/link_prediction/WN18RR_v2/SNRI/epoch\=0-Eval\|auc\=0.987.ckpt

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
    --num_layers $NUM_LAYERS \
    --num_bases $NUM_BASES \
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
    --dropout $DROPOUT \
    --calc_hits $CALC_HITS \
    --checkpoint_dir $CHECKPOINT_DIR \
    --inductive \
    --test_only \
    