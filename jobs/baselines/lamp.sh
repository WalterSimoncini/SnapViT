# Pruning parameters
MIN_HEAD_RATIO=0.2
MIN_HIDDEN_DIM_RATIO=0.05

# Prunability score estimation parameters
PRUNING_STRATEGY=lamp
NUM_SAMPLES=1000
ESTIMATION_EPOCHS=1
BATCH_SIZE=16
NUM_WORKERS=8

# Pruning and evaluation datasets
PRUNING_DATASET_SPLIT=train
PRUNING_DATASET_NAME=imagenet-1k

EVAL_DATASET_NAMES="textures aircraft eurosat cifar10 cifar100 pets imagenet-1k"

# Genetic algorithm parameters
CACHE_DIR=/home/$USER/scratch/cache
OUTPUT_DIR=/home/$USER/scratch/baselines/lamp

mkdir -p $OUTPUT_DIR

# Names of the model families to prune
MODEL_VALUES=(
    "dino_vitb16"
)

MLP_PRUNING_RATIOS=(0.15 0.25 0.35 0.45 0.55 0.65)
HEAD_PRUNING_RATIOS=(0 0.1 0.2 0.3 0.4 0.5)

# This loop will run model x ratios iterations
for MODEL in ${MODEL_VALUES[@]}; do
    for i in "${!MLP_PRUNING_RATIOS[@]}"; do
        MLP_PRUNING_RATIO=${MLP_PRUNING_RATIOS[$i]}
        HEAD_PRUNING_RATIO=${HEAD_PRUNING_RATIOS[$i]}

        python prune-baseline.py \
            --pruning-dataset $PRUNING_DATASET_NAME \
            --pruning-dataset-split $PRUNING_DATASET_SPLIT \
            --eval-datasets $EVAL_DATASET_NAMES \
            --cache-dir $CACHE_DIR \
            --seed 42 \
            --batch-size $BATCH_SIZE \
            --num-workers $NUM_WORKERS \
            --max-samples $NUM_SAMPLES \
            --model-type $MODEL \
            --pruning-ratio $MLP_PRUNING_RATIO \
            --pruning-ratio-heads $HEAD_PRUNING_RATIO \
            --num-estimation-epochs $ESTIMATION_EPOCHS \
            --pruning-strategy $PRUNING_STRATEGY \
            --min-hidden-dim-ratio $MIN_HIDDEN_DIM_RATIO \
            --min-head-ratio $MIN_HEAD_RATIO \
            --output-dir $OUTPUT_DIR \
            --save-pruned-model \
            --save-features
    done
done