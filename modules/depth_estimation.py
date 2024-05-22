
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class DepthEstimatorBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, image):
        return None

class DepthEstimatorConstant(DepthEstimatorBase):
    def __init__(self, multiplier=10) -> None:
        super().__init__()
        self.multiplier = multiplier

    def __call__(self, image, mask):
        return self.multiplier * torch.ones_like(image[0, ...])[None, ...]
