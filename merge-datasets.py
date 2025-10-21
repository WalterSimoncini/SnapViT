import os
import PIL
import shutil
import argparse

from typing import List
from datasets import DatasetDict, Dataset

from src.utils.logging import configure_logging
from src.utils.misc import seed_everything, default_cache_dir
from src.datasets import load_dataset, DatasetSplit, DatasetType


def create_subset_generator(samples: List[PIL.Image.Image]):
    def generator():
        for sample in samples:
            yield {
                "image": sample
            }

    return generator


def main(args):
    configure_logging()
    seed_everything(args.seed)

    # Delete the target directory if it exists already
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)

    samples = []

    for dataset_type in args.datasets:
        dataset = load_dataset(
            type_=dataset_type,
            split=DatasetSplit.TRAIN,
            cache_dir=args.cache_dir,
            transform=None
        )

        samples.extend([dataset[i][0] for i in range(args.num_samples)])

    processed_dataset_dict = {
        "train": Dataset.from_generator(
            create_subset_generator(samples=samples),
            num_proc=args.num_proc
        )
    }

    out_dataset = DatasetDict(processed_dataset_dict)
    out_dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick samples from multiple datasets and merge them into a single one")

    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")
    parser.add_argument("--cache-dir", type=str, required=False, default=default_cache_dir(), help="The cache directory for datasers and models")

    parser.add_argument("--num-proc", type=int, default=18, help="Number of processors used for generating the dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save the merged dataset")
    parser.add_argument("--datasets", type=DatasetType, nargs='+', choices=list(DatasetType), default=[DatasetType.IN1K], help="The datasets to pick samples from")
    parser.add_argument("--num-samples", type=int, default=None, help="The number of samples to pick from each dataset")

    main(parser.parse_args())
