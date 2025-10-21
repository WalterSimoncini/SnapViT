CACHE_DIR=/home/$USER/scratch/cache

# Create a data directory for the shaders dataset
mkdir $CACHE_DIR/shaders21k

# Download the dataset
wget -P $CACHE_DIR/shaders21k/ http://data.csail.mit.edu/synthetic_training/shaders21k/zipped_data/shaders21k.zip

# Unzip the data
unzip $CACHE_DIR/shaders21k/shaders21k.zip -d $CACHE_DIR/shaders21k
