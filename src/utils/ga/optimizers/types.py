import torch
import numpy as np

from typing import Callable, Union, List


FitnessFunction = Callable[[Union[np.ndarray, List[float], torch.Tensor]], float]
