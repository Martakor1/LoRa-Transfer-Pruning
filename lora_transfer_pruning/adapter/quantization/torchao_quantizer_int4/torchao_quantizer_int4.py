from __future__ import annotations

from collections import Counter
from typing import cast

import torch
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from torch import nn
from transformers import PreTrainedModel

from lora_transfer_pruning.adapter.quantization.universal_dequantizer import UniversalDequantizer
from lora_transfer_pruning.port.model_quantizer import LayerFilter, ModelQuantizer

from .torchao_int4_linear import TorchAOInt4Linear
from .torchao_repacker_int4 import TorchAORepackerInt4


class TorchAOQuantizerInt4(ModelQuantizer):
    """Convert dense or GPTQ projections to TorchAO INT4."""

    def __init__(self, group_size: int = 128) -> None:
        # The experiment uses groups of 128. Other checkpoints may need smaller groups.
        if group_size not in (32, 64, 128, 256):
            raise ValueError("HQQ group_size must be 32, 64, 128, or 256")
        self._group_size = group_size
        self._repacker = TorchAORepackerInt4(group_size)

    @torch.no_grad()
    def quantize(
        self,
        model: PreTrainedModel,
        *,
        device: torch.device | str | None = None,
        filter_fn: LayerFilter | None = None,
    ) -> PreTrainedModel:
        """Replace eligible layers in place, then return the prepared model."""
        if getattr(model, "peft_config", None):
            raise TypeError("Quantize the base model before adding LoRA")
        placement = self._quantization_device(model, device)
        targets, gptq_paths = self._select_target_layers(model, filter_fn)
        self._fp_type_check(model, targets, placement)
        self._quantize(model, targets, placement)

        if placement is not None and not getattr(model, "hf_device_map", None):
            cast(nn.Module, model).to(placement)
        if gptq_paths and len(targets) == len(gptq_paths):
            self._clear_gptq_metadata(model)
        return model

    def _select_target_layers(
        self, model: PreTrainedModel, filter_fn: LayerFilter | None,
    ) -> tuple[list[str], list[str]]:
        '''Select simple linears layers for quantization. Our quantized GPTQ layers.'''
        output_head = model.get_output_embeddings()
        head_modules = set(output_head.modules()) if output_head is not None else set()
        gptq_paths = self._gptq_linear_paths(model)
        if gptq_paths:
            targets = []
            for path in gptq_paths:
                layer = model.get_submodule(path)
                if layer not in head_modules and (filter_fn is None or filter_fn(layer, path)):
                    targets.append(path)
        else:
            targets = self._dense_linear_paths(model, head_modules, filter_fn)
        if not targets:
            raise ValueError("No eligible linear weights for TorchAO INT4 quantization")
        return targets, gptq_paths

    def _fp_type_check(
        self, model: PreTrainedModel, targets: list[str], placement: torch.device | None,
    ) -> None:
        for path in targets:
            layer = cast("nn.Linear | TorchLinear", model.get_submodule(path))
            if isinstance(layer, TorchLinear):
                storage: torch.Tensor | None = getattr(layer, "scales", None)
                if storage is None or any(getattr(layer, name, None) is None for name in ("qweight", "qzeros", "g_idx")):
                    raise ValueError(f"Missing packed GPTQ weights for {path}")
            else:
                storage = layer.weight
            if (
                storage.is_meta
                or (placement or storage.device).type != "cuda"
                or storage.dtype not in (torch.float16, torch.bfloat16)
                or layer.in_features % self._group_size
            ):
                raise ValueError(
                    f"HQQ INT4 requires CUDA FP16/BF16 weights and in_features divisible "
                    f"by {self._group_size}: {path}"
                )

    def _quantize(
        self, model: PreTrainedModel, targets: list[str], placement: torch.device | None,
    ) -> None:
        # Process one layer at a time, including checkpoints initially loaded on CPU.
        for path in targets:
            layer = cast("nn.Linear | TorchLinear", model.get_submodule(path))
            if isinstance(layer, TorchLinear):
                scales: torch.Tensor | None = getattr(layer, "scales", None)
                assert scales is not None  # Checked for all targets before conversion.
                packed = self._convert_gptq_linear(layer, placement or scales.device)
            else:
                packed = self._quantize_universal_module_weight(layer, placement or layer.weight.device, layer.bias)
            model.set_submodule(path, packed.train(layer.training))

    @staticmethod
    def _quantization_device( #TODO: make simplier? think about universal device placement?
        model: PreTrainedModel, device: torch.device | str | None,
    ) -> torch.device | None:
        if device is None:
            return None
        placement = torch.device(device)
        device_map = getattr(model, "hf_device_map", None)
        if device_map:
            if placement.type == "cuda" and placement.index is None:
                placement = torch.device("cuda", torch.cuda.current_device())
            for mapped in device_map.values():
                if mapped == "disk":
                    raise ValueError("Load the model without disk offload before quantization")
                mapped_device = torch.device("cuda", mapped) if isinstance(mapped, int) else torch.device(mapped)
                if mapped_device.type == "cuda" and mapped_device.index is None:
                    mapped_device = torch.device("cuda", torch.cuda.current_device())
                if mapped_device != placement:
                    raise ValueError("Requested quantization device does not match the loaded model's device map")
        return placement

    @staticmethod
    def _gptq_linear_paths(model: PreTrainedModel) -> list[str]:
        paths = []
        for path, layer in model.named_modules():
            if isinstance(layer, BaseQuantLinear):
                if not isinstance(layer, TorchLinear):
                    raise TypeError(
                        f"Cannot convert {path} ({type(layer).__name__}). Load it with the GPTQ_TORCH backend."
                    )
                paths.append(path)
        return paths

    @staticmethod
    def _dense_linear_paths(
        model: PreTrainedModel,
        head_modules: set[nn.Module],
        filter_fn: LayerFilter | None,
    ) -> list[str]:
        parameter_uses = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
        targets = []
        for path, layer in model.named_modules():
            if type(layer) is not nn.Linear or layer in head_modules:
                continue
            if type(layer.weight) is not nn.Parameter or parameter_uses[id(layer.weight)] > 1:
                continue
            if filter_fn is None or filter_fn(layer, path):
                targets.append(path)
        return targets

    def _convert_gptq_linear(self, layer: TorchLinear, device: torch.device) -> TorchAOInt4Linear:
        bias = layer.bias
        if bias is not None and not isinstance(bias, nn.Parameter):
            bias = nn.Parameter(bias.to(device), requires_grad=False)
        tensors = self._repacker.try_repack(layer, device)
        if tensors is not None:
            return TorchAOInt4Linear.from_packed(tensors, device=device, bias=bias)
        return self._quantize_universal_module_weight(layer, device, bias)

    def _quantize_universal_module_weight(
        self, layer: nn.Linear | TorchLinear, device: torch.device, bias: nn.Parameter | None,
    ) -> TorchAOInt4Linear:
        weight = UniversalDequantizer.to_dense_weight(layer, device)
        return TorchAOInt4Linear(weight, group_size=self._group_size, bias=bias)

    @staticmethod
    def _clear_gptq_metadata(model: PreTrainedModel) -> None:
        """Drop GPTQ loader metadata once its packed layers have been replaced."""
        if hasattr(model.config, "quantization_config"):
            delattr(model.config, "quantization_config")
        if "is_quantized" in vars(model):
            del model.is_quantized
        if getattr(model, "quantization_method", None) is not None:
            setattr(model, "quantization_method", None)
        if getattr(model, "hf_quantizer", None) is not None:
            setattr(model, "hf_quantizer", None)
