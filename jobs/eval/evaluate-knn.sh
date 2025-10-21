# This script loads a precomputed set of features and uses them
# to train and evaluate a knn classifier.
MODEL_TYPE=dino_vitb16

DATASETS=(textures aircraft eurosat cifar10 cifar100 pets imagenet-1k)

BASE_DIR=/home/$USER/scratch/pruning/artifacts/ga/

for FOLDER in $BASE_DIR/*; do
    for SPARSITY_FOLDER in $FOLDER/*; do
        if [ -d "$SPARSITY_FOLDER" ]; then
            BASE_FEATURES_DIR=$SPARSITY_FOLDER/pruned/features/
            OUTPUT_DIR=$SPARSITY_FOLDER/pruned/features/knn

            mkdir -p $OUTPUT_DIR

            for DATASET in ${DATASETS[@]}; do
                DATASET_OUTPUT_DIR=$OUTPUT_DIR/$DATASET

                mkdir -p $DATASET_OUTPUT_DIR

                python evaluate-knn.py \
                    --seed 42 \
                    --dataset-path $BASE_FEATURES_DIR/$DATASET.pt \
                    --output-dir $DATASET_OUTPUT_DIR
            done
        fi
    done
done