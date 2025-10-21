# This script loads a precomputed set of features and uses them
# to train and evaluate a linear classifier.

# The same configuration as DINOv1. See
# https://github.com/facebookresearch/dino/blob/main/eval_linear.py
PATIENCE=100
MAX_EPOCHS=100

BATCH_SIZE=256
NUM_WORKERS=16

LEARNING_RATES="0.001"
WEIGHT_DECAYS="0"

DATASETS=(textures aircraft eurosat cifar10 cifar100 pets imagenet-1k)

BASE_DIR=/home/$USER/scratch/pruning/artifacts/ga

for FOLDER in $BASE_DIR/*; do
    for SPARSITY_FOLDER in $FOLDER/*; do
        if [ -d "$SPARSITY_FOLDER" ]; then
            BASE_FEATURES_DIR=$SPARSITY_FOLDER/pruned/features/
            OUTPUT_DIR=$SPARSITY_FOLDER/pruned/features/linear

            mkdir -p $OUTPUT_DIR

            for DATASET in ${DATASETS[@]}; do
                DATASET_OUTPUT_DIR=$OUTPUT_DIR/$DATASET

                mkdir -p $DATASET_OUTPUT_DIR

                python evaluate-linear.py \
                    --seed 42 \
                    --dataset-path $BASE_FEATURES_DIR/$DATASET.pt \
                    --batch-size $BATCH_SIZE \
                    --num-workers $NUM_WORKERS \
                    --max-epochs $MAX_EPOCHS \
                    --patience $PATIENCE \
                    --learning-rates $LEARNING_RATES \
                    --weight-decays $WEIGHT_DECAYS \
                    --output-dir $DATASET_OUTPUT_DIR
            done
        fi
    done
done