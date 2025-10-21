NUM_PROC=8
CACHE_DIR=/home/$USER/scratch/cache

# Remove the data directory if it exists
rm -rf $CACHE_DIR/pets
mkdir $CACHE_DIR/pets

# Download the images and annotations
wget -P $CACHE_DIR/pets https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
wget -P $CACHE_DIR/pets https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz

# Unpack the images and annotations
tar -xf $CACHE_DIR/pets/images.tar.gz -C $CACHE_DIR/pets
tar -xf $CACHE_DIR/pets/annotations.tar.gz -C $CACHE_DIR/pets

# Generate the dataset
python pets.py \
    --num-proc 18 \
    --cache-dir $CACHE_DIR \
    --root-dir $CACHE_DIR/pets/ \
    --output-folder $CACHE_DIR/pets-hf
