import os
import sys
import time
import json
import torch
import logging
import argparse
import torch.nn as nn

from typing import Optional

from src.utils.misc import serialize_args
from src.models.prunable import PrunableModel


def configure_logging():
    logging.basicConfig(
        format="[%(asctime)s:%(levelname)s]: %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{int(time.time())}.out"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def save_run_info(
    output_dir: str,
    metrics: dict,
    args: argparse.Namespace,
    model: Optional[nn.Module | PrunableModel] = None
):
    try:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            f.write(json.dumps(metrics, indent=4))
    except Exception as e:
        logging.error(f"error saving the metrics to disk: {e}")
        logging.error(f"the metrics were: {metrics}")

    try:
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            f.write(json.dumps(serialize_args(args=args), indent=4))
    except Exception as e:
        logging.error(f"error saving the args to disk: {e}")
        logging.error(f"the args were: {args}")

    # Save the model to disk if it's not None
    if model is not None:
        save_pruned_model(output_dir=output_dir, model=model)


def save_pruned_model(
    output_dir: str,
    model: nn.Module | PrunableModel
):
    model = model.model if isinstance(model, PrunableModel) else model

    try:
        torch.save(model.state_dict(), os.path.join(output_dir, "pruned-model.pt"))

        logging.info(f"saved the pruned model to {output_dir}")
    except Exception as e:
        logging.error(f"error saving the pruned model to disk: {e}")
