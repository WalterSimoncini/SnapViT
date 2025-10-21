CACHE_DIR=/home/$USER/scratch/cache

# Generate a dataset with 1001 images, equally sampled from each input dataset
python merge-datasets.py \
    --datasets textures aircraft eurosat cifar10 cifar100 pets imagenet-1k \
    --output-dir $CACHE_DIR/merged \
    --num-samples 143 \
    --seed 42
