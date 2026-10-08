from dataclasses import dataclass, field

from transformer_lens.tools.analysis.attribution_patching import GradientCache
from .activation_corruption import ActivationCorruption

@dataclass
class PruningTrace(GradientCache):
    corruptions: dict[str, ActivationCorruption]
    # These hooks may still have known TL conversion defects.
    unresolved_pruning_hooks: set[str] = field(default_factory=set)


    def to_cpu(self) -> "PruningTrace":
        for name, tensor in self.activations.items():
            self.activations[name] = tensor.detach().cpu()
        for name, tensor in self.gradients.items():
            if tensor is not None:
                self.gradients[name] = tensor.detach().cpu()
        self.metric=self.metric.detach().cpu()
        return self