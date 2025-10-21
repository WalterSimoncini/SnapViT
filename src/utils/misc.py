import os
import pwd
import argparse

import torch
import random
import numpy as np

from enum import Enum
from typing import Union
from torch.nn.functional import pad


def get_device() -> torch.device:
    # Use MPS if we're running on an Apple Silicon Mac
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def default_cache_dir() -> str:
    username = pwd.getpwuid(os.getuid())[0]

    return f"/home/{username}/scratch/cache"


def seed_everything(seed: int):
	random.seed(seed)
	np.random.seed(seed)

	os.environ["PYTHONHASHSEED"] = str(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def pad_vector_to_match(
	x: torch.Tensor,
    target_length: int,
	value: Union[int, float]
) -> torch.Tensor:
	pad_length = target_length - x.shape[0]

	if pad_length > 0:
		# (left padding, right padding)
		return pad(x, (0, pad_length), "constant", value)

	return x


def serialize_args(args: argparse.Namespace) -> dict:
    """
		Converts an argparse Namespace to a python dictionary
		that can be serialized to JSON.
	"""
    serialized_args = vars(args)

    for k, v in serialized_args.items():
        if isinstance(v, Enum):
            serialized_args[k] = v.value
        elif isinstance(v, list):
            serialized_args[k] = [x.value if isinstance(x, Enum) else x for x in v]

    return serialized_args
