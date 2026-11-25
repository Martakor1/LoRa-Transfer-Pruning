from torch_pruning.pruner.function import BasePruningFunc

class PackedLinearDepGraphOnlyPruner(BasePruningFunc):
    def get_in_channels(self, layer):
        return layer.in_features

    def get_out_channels(self, layer):
        return layer.out_features

    def prune_in_channels(self, layer, idxs):
        raise RuntimeError(
            "Packed linear layers support dependency tracing only. Use simulated pruning."
        )

    def prune_out_channels(self, layer, idxs):
        raise RuntimeError(
            "Packed linear layers support dependency tracing only. Use simulated pruning."
        )
