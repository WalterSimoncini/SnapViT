# Elastic ViTs from Pretrained Models without Retraining - NeurIPS 2025

[Walter Simoncini](https://walter.ashita.nl/)<sup>1,2,*</sup>, [Michael Dorkenwald](https://mdorkenwald.com/)<sup>2,*</sup>, [Tijmen Blankevoort](https://scholar.google.com/citations?user=OGEyrG8AAAAJ&hl=en)<sup>3</sup>, [Cees G.M. Snoek](https://www.ceessnoek.info/)<sup>2</sup>, [Yuki M. Asano](https://yukimasano.github.io/)<sup>1</sup>.

1. University of Technology Nuremberg.
2. University of Amsterdam.
3. NVIDIA.

\* Indicates equal contribution.

This repository contains the code for our paper [Elastic ViTs from Pretrained Models without Retraining](https://elastic.ashita.nl/).

> Vision foundation models achieve remarkable performance but are only available in a limited set of pre-determined sizes, forcing sub-optimal deployment choices under real-world constraints. We introduce a new post-pretraining structured pruning method that enables elastic inference across a continuum of compute budgets. Our approach efficiently combines gradient information with cross-network structure correlations, approximated via an evolutionary algorithm, does not require labeled data, generalizes to models without a classification head, and is retraining-free. Experiments on DINO, SigLIPv2, DeIT, and AugReg models demonstrate superior performance over state-of-the-art methods across various sparsities, requiring less than five minutes on a single A100 GPU to generate elastic models that can be adjusted to any computational budget. Our key contributions include an efficient pruning strategy for pretrained Vision Transformers, a novel evolutionary approximation of Hessian off-diagonal structures, and a self-supervised importance scoring mechanism that maintains strong performance without requiring retraining or labels.

## Getting Started

This code was developed using `Python 3.11`, so to run the experiments, we recommend that you create an appropriate environment and install the required dependencies as follows:

```bash
conda create -n vit-pruning python=3.11
conda activate vit-pruning

# Install the required Python libraries
pip install -r requirements.txt
```

## Dataset Setup

Most datasets are loaded automatically on demand and require no special setup. The only exceptions are:

- ImageNet-1k: see the [torchvision instructions](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html) on how to download and configure the dataset.
- Oxford-IIT Pets: use the `jobs/datasets/pets.sh` script to download the images and create a HuggingFace dataset. Don't forget to configure the `NUM_PROC` and `CACHE_DIR` variables beforehand.
- Shaders21k: this dataset is only used in an ablation study, but you can nonetheless download it by running the `jobs/datasets/shaders.sh` script.
- Merged Data: as above, the `merged` dataset is only used for an ablation in our paper. You can re-create it by running `jobs/datasets/merged.sh`. Using this dataset should lead to better generalization for models trained on large-scale datasets.

## Model Setup

Models are downloaded automatically on demand from HuggingFace, with the exception of DINOv3, for which you'll have to request a download link via [this form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and download models as follows:

```sh
wget -P /home/$USER/scratch/cache/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth <VIT-B/16 URL>
wget -P /home/$USER/scratch/cache/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth <VIT-H+/16 URL>
```

## Reproducing Experiments

Some of the core experiments in the paper can be reproduced as-is using the scripts in the `jobs` directory. Scripts generally assume that you have a dataset cache folder rooted at `/home/$USER/scratch/cache`; make sure to change it as appropriate.

### Baselines (LAMP, SNIP Magnitude)

We provide scripts to evaluate other state-of-the-art baselines implemented in our framework.

- `jobs/baselines/lamp.sh`: prune DINO ViT-B/16 using the LAMP score as the prunability score.
- `jobs/baselines/lamp.sh`: prune DINO ViT-B/16 using the SNIP Magnitude score as the prunability score.

### Elastic Pruning

We provide two scripts to perform elastic pruning:

- `jobs/ga/dino.sh`: perform elastic pruning for a DINO ViT-B/16 backbone.
- `jobs/ga/dino-sparsegpt.sh`: same as above, but with a post-pruning SparseGPT-style weight correction step.

### Evaluation Utilities

We provide three utility scripts to facilitate evaluation:

- `jobs/linear/extract.sh`: encode features for several `(model, dataset)` combinations. Useful to extract features from unpruned models that can be used for evaluation. The `extract.py` script can also be used to extract features from a pruned model by specifying the `--pruned-checkpoint-path` and NOT specifying the `--eval-baseline` flag.
- `jobs/linear/evaluate-knn.sh`: given the path to on-disk encoded features, runs a k-nn evaluation and saves the top-1 accuracy metric on disk.
- `jobs/linear/evaluate-linear.sh`: given the path to on-disk encoded features, runs a linear probing evaluation and saves the top-1 accuracy metric on disk. We use the same initialization and parameters as the [DINOv1 linear evaluation](https://github.com/facebookresearch/dino/blob/main/eval_linear.py).

The file containing encoded features is expected to have the following structure:

```json
{
    "train": {
        "features": [N, E] torch.Tensor,
        "targets": [N] torch.Tensor
    },
    "valid": {
        "features": [N, E] torch.Tensor,
        "targets": [N] torch.Tensor
    },
    "test": {
        "features": [N, E] torch.Tensor,
        "targets": [N] torch.Tensor
    },
    "merge_train_valid": bool
}
```

The `merge_train_valid` flag indicates whether the training and validation sets should be merged for the k-nearest neighbor and linear evaluation. The flag is true for datasets whose validation split was generated as an 80/20 split of the training set.

## Inference

Once you have generated importance scores via `prune.py` (with `--save-importance-scores`), or downloaded the pre-computed ones, you can use `inference.py` to perform elastic inference at any target sparsity:

```bash
python inference.py \
    --model-type dino_vitb16 \
    --importance-scores-path /path/to/importance-scores.pt \
    --mlp-pruning-ratio 0.35 \
    --head-pruning-ratio 0.2 \
    --eval-datasets imagenet-1k cifar10
```

> [!TIP]
> We provide pre-computed importance scores in the `scores` directory for the following models: AugReg ViT-B/16, AugReg ViT-L/16, DeIT ViT-B/16, DeIT-3 ViT-S/16, DeIT-3 ViT-B/16, DeIT-3 ViT-L/16, DeIT-3 ViT-H/14, DINO ViT-B/16, DINOv3 ViT-B/16, and SigLIP2 ViT-B/16.

The `scores/imagenet-1k` and `scores/merged` directories contain scores estimated using the `ImageNet-1k` and `merged` datasets respectively.

### SparseGPT Weight Correction

For improved accuracy, you can enable SparseGPT-based weight correction on the `fc2` and `attn.proj` layers:

```bash
python inference.py \
    --model-type dino_vitb16 \
    --importance-scores-path /path/to/importance-scores.pt \
    --mlp-pruning-ratio 0.35 \
    --head-pruning-ratio 0.2 \
    --apply-correction \
    --correction-dataset imagenet-1k \
    --correction-max-samples 1000 \
    --eval-datasets imagenet-1k
```

#### DINOv3

DINOv3 models have extreme Hessian outliers that require different damping settings for stable weight correction. Use `--correction-damping-strategy max` and `--correction-damping-percentage 0.1`:

```bash
python inference.py \
    --model-type dinov3_vitb16 \
    --importance-scores-path /path/to/importance-scores.pt \
    --mlp-pruning-ratio 0.35 \
    --head-pruning-ratio 0.2 \
    --apply-correction \
    --correction-dataset imagenet-1k \
    --correction-max-samples 1000 \
    --correction-damping-strategy max \
    --correction-damping-percentage 0.1 \
    --eval-datasets imagenet-1k
```

### Programmatic Usage

You can also use the `ElasticViT` class directly in your own code for dynamic inference at different sparsities:

```python
from src.models import load_model, ModelType
from src.inference import ElasticImportanceScores, ElasticViT

# Load importance scores and create elastic model
scores = ElasticImportanceScores.load("importance-scores.pt", device=device)
elastic = ElasticViT(
    model_factory=lambda: load_model(ModelType.DINO_VIT_B_16, cache_dir, device)[0],
    scores=scores,
    device=device
)

# Prune to target sparsity
elastic.prune(mlp_pruning_ratio=0.35, head_pruning_ratio=0.2)
output = elastic(images)

# Prune further (no reset needed when pruning more aggressively)
elastic.prune(mlp_pruning_ratio=0.5, head_pruning_ratio=0.4)

# Prune to a shallower sparsity (reset is called automatically)
elastic.prune(mlp_pruning_ratio=0.2, head_pruning_ratio=0.1)
```

## Results

We provide benchmark results for pruned models at various sparsity levels in [`results/in1k.md`](results/in1k.md).

## Reference

If you found our work useful please cite us as follows:

```
@inproceedings{simoncini2025elastic,
  title={Elastic ViTs from Pretrained Models without Retraining},
  author={Walter Simoncini and Michael Dorkenwald and Tijmen Blankevoort and Cees G.M. Snoek and Yuki M. Asano},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=OU6FXkSIe0}
}
```
