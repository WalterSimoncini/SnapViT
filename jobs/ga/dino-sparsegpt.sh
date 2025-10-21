MIN_HEAD_RATIO=0.2
MIN_HIDDEN_DIM_RATIO=0.05

# Prunability score estimation parameters
PRUNING_STRATEGY=sparse-gpt-dino
NUM_SAMPLES=1000
ESTIMATION_EPOCHS=1
BATCH_SIZE=16
NUM_WORKERS=8

# Pruning and evaluation datasets
PRUNING_DATASET_SPLIT=train
PRUNING_DATASET_NAME=imagenet-1k

EVAL_DATASET_NAMES="textures aircraft eurosat cifar10 cifar100 pets imagenet-1k"

# Genetic algorithm parameters
GA_OPTIMIZER=xnes
GA_BATCH_SIZE=64
GA_NUM_PCA_COMPONENTS=192
GA_MAX_EVAL_SAMPLES=1000
GA_MAX_FUNCTION_EVALUATIONS=50
GA_OPTIMIZATION_DATASET=imagenet-1k
GA_OPTIMIZATION_DATASET_SPLIT=train
GA_LOSS_TYPES=cosine-similarity

# Weight correction parameters
CORRECTION_DATASET=imagenet-1k
CORRECTION_MAX_SAMPLES=1000

CACHE_DIR=/home/$USER/scratch/cache
OUTPUT_DIR=/home/$USER/scratch/pruning/artifacts/ga/sparsegpt

mkdir -p $OUTPUT_DIR

# Names of the model families to prune
MODEL_VALUES=(
    "dino_vitb16"
)

MLP_PRUNING_RATIOS="0.15 0.35 0.55 0.65"
HEAD_PRUNING_RATIOS="0 0.2 0.4 0.5"

EVAL_MLP_PRUNING_RATIOS="0.15 0.25 0.35 0.45 0.55 0.65"
EVAL_HEAD_PRUNING_RATIOS="0 0.1 0.2 0.3 0.4 0.5"

# This loop will run model x ratios iterations
for MODEL in ${MODEL_VALUES[@]}; do
    python prune.py \
        --pruning-dataset $PRUNING_DATASET_NAME \
        --pruning-dataset-split $PRUNING_DATASET_SPLIT \
        --eval-datasets $EVAL_DATASET_NAMES \
        --cache-dir $CACHE_DIR \
        --seed 42 \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --max-samples $NUM_SAMPLES \
        --model-type $MODEL \
        --num-estimation-epochs $ESTIMATION_EPOCHS \
        --pruning-strategy $PRUNING_STRATEGY \
        --min-hidden-dim-ratio $MIN_HIDDEN_DIM_RATIO \
        --min-head-ratio $MIN_HEAD_RATIO \
        --ga-max-eval-samples $GA_MAX_EVAL_SAMPLES \
        --ga-optimization-dataset $GA_OPTIMIZATION_DATASET \
        --ga-optimization-dataset-split $GA_OPTIMIZATION_DATASET_SPLIT \
        --ga-keep-in-memory \
        --ga-batch-size $GA_BATCH_SIZE \
        --ga-optimizer $GA_OPTIMIZER \
        --ga-max-function-evaluations $GA_MAX_FUNCTION_EVALUATIONS \
        --ga-num-pca-components $GA_NUM_PCA_COMPONENTS \
        --ga-loss-types $GA_LOSS_TYPES \
        --ga-mlp-pruning-ratios $MLP_PRUNING_RATIOS \
        --ga-heads-pruning-ratios $HEAD_PRUNING_RATIOS \
        --eval-mlp-pruning-ratios $EVAL_MLP_PRUNING_RATIOS \
        --eval-heads-pruning-ratios $EVAL_HEAD_PRUNING_RATIOS \
        --output-dir $OUTPUT_DIR \
        --save-pruned-model \
        --ga-save-weights \
        --save-features \
        --correction-dataset $CORRECTION_DATASET \
        --correction-max-samples $CORRECTION_MAX_SAMPLES
done