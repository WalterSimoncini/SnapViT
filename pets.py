"""
    This script generates a HuggingFace dataset for
    the Oxford-IIT Pets dataset
"""
import os
import datasets
import argparse

from PIL import Image
from typing import List
from datasets import Dataset, Features, DatasetDict

from src.utils.logging import configure_logging


def generate_split(images_base_path: str, annotations_path: List[str]):
    """
        Given the path to the folder containing the dataset images
        and the path to an annotation file containing space-separated
        triplets in the format "Image CLASS-ID SPECIES BREED ID" generates
        rows for an HuggingFace dataset containing images and their class
    """
    samples = open(annotations_path).read().split("\n")

    for sample in samples:
        if sample.strip() == "":
            # Skip blank lines
            continue

        sample_metadata = sample.split(" ")
        image_path, label = sample_metadata[:2]
        image_path = f"{image_path}.jpg"

        yield {
            "image": Image.open(os.path.join(images_base_path, image_path)),
            "label": int(label) - 1
        }


def main(args):
    configure_logging()

    images_folder = os.path.join(args.root_dir, "images")

    test_annotations_path = os.path.join(args.root_dir, "annotations", "test.txt")
    # Here we use trainval for the training dataset as the original paper states
    # "Algorithms are trained on the training and validation subsets and tested
    # on the test subset"
    train_annotations_path = os.path.join(args.root_dir, "annotations", "trainval.txt")

    features = Features({
        "image": datasets.Image(),
        # The pets dataset has 37 classes https://www.robots.ox.ac.uk/~vgg/data/pets/
        "label": datasets.ClassLabel(num_classes=37)
    })

    out_dataset = DatasetDict({
        "train": Dataset.from_generator(
            num_proc=args.num_proc,
            generator=generate_split,
            features=features,
            cache_dir=args.cache_dir,
            gen_kwargs={
                "images_base_path": images_folder,
                "annotations_path": train_annotations_path
            }
        ),
        "test": Dataset.from_generator(
            num_proc=args.num_proc,
            generator=generate_split,
            features=features,
            cache_dir=args.cache_dir,
            gen_kwargs={
                "images_base_path": images_folder,
                "annotations_path": test_annotations_path
            }
        ),
    })

    out_dataset.save_to_disk(args.output_folder, num_proc=args.num_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a HuggingFace dataset for the Oxford-IIT Pets dataset")

    parser.add_argument("--num-proc", type=int, default=18, help="Number of processors used for generating and saving the dataset")
    parser.add_argument("--output-folder", type=str, required=True, help="Where to save the processed dataset")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")
    parser.add_argument("--root-dir", type=str, required=True, help="The root directory where the 'images' and 'annotations' folders are stored")

    main(parser.parse_args())
