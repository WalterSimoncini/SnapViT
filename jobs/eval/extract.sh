# This script extract features for a set of datasets and pretrained
# models. Currently, only the extraction of base (unpruned) models
# is supported.

CACHE_DIR=/home/$USER/scratch/cache

BATCH_SIZE=64
NUM_WORKERS=8

MODELS=(dino_vitb16 augreg-vit-b-16-in21k-ft-in1k)
DATASETS=(textures aircraft eurosat cifar10 cifar100 pets imagenet-1k)

# Extract features for the base (unpruned) models
for MODEL_TYPE in ${MODELS[@]}; do
    OUTPUT_DIR=/home/$USER/scratch/pruning/artifacts/unpruned/features/$MODEL_TYPE

    mkdir -p $OUTPUT_DIR

    for DATASET in ${DATASETS[@]}; do
        python extract.py \
            --seed 42 \
            --model-type $MODEL_TYPE \
            --cache-dir $CACHE_DIR \
            --dataset $DATASET \
            --batch-size $BATCH_SIZE \
            --num-workers $NUM_WORKERS \
            --output-dir $OUTPUT_DIR \
            --eval-baseline
    done
done