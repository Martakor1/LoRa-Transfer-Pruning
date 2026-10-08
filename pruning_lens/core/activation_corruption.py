from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

@dataclass
class Defect(ABC):
    restore_info: Any
    
    @abstractmethod
    def restore_activation(self, activation: torch.Tensor) -> torch.Tensor:
        pass
    
@dataclass
class RemovedChannelsDefect(Defect):
    '''List of removed channels'''
    restore_info: torch.Tensor
    
    def restore_activation(self, activation: torch.Tensor) -> torch.Tensor:
        channels_was = activation.shape[-1] + self.restore_info.numel()
        idxs = self.restore_info.to(device=activation.device, dtype=torch.long)
        if ((idxs < 0) | (idxs >= channels_was)).any():
            raise ValueError(f"Removed channel index is outside the original width {channels_was}")
        keep_mask = torch.ones(channels_was, dtype=torch.bool, device=activation.device)
        keep_mask[idxs] = False
        restored_activation = torch.zeros(activation.shape[:-1] + (channels_was,), device=activation.device, dtype=activation.dtype)
        restored_activation[..., keep_mask] = activation
        return restored_activation
    
@dataclass
class ReshapeDefect(Defect):
    '''Initial shape of the activation before reshaping'''
    restore_info: tuple[int, ...]
    
    def restore_activation(self, activation: torch.Tensor) -> torch.Tensor:
        return activation.reshape(self.restore_info)


@dataclass
class TransposeDefect(Defect):
    '''Axes swapped by the conversion; swapping again restores the layout.'''
    restore_info: tuple[int, int]

    def restore_activation(self, activation: torch.Tensor) -> torch.Tensor:
        return activation.transpose(*self.restore_info)


@dataclass
class RearrangeDefect(Defect):
    '''Inverse einops pattern and a snapshot of its required axis lengths.'''
    restore_info: tuple[str, dict[str, int]]

    def restore_activation(self, activation: torch.Tensor) -> torch.Tensor:
        pattern, axes_lengths = self.restore_info
        return rearrange(activation, pattern, **axes_lengths)

@dataclass
class ActivationCorruption:
    '''Defects in order, they was applied to the activation'''
    defects: list[Defect]
    
    def restore_activation(self, activation: torch.Tensor) -> torch.Tensor:
        for defect in reversed(self.defects):
            activation = defect.restore_activation(activation)
        return activation
