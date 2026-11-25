from collections.abc import Generator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock
from typing import Any, cast
from unittest.mock import patch

from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from peft.import_utils import is_transformers_ge_v5
from torch import nn
from transformers import PreTrainedModel

from lora_transfer_pruning.port.lora_injector import LoraInjector


_expert_conversion_lock = Lock()


class PeftLoraInjector(LoraInjector):
    def __init__(self, custom_modules: Mapping[type[nn.Module], type[nn.Module]] | None = None) -> None:
        self._custom_modules = dict(custom_modules or {})

    def add_lora(
        self,
        model: PreTrainedModel,
        config: LoraConfig,
        *,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
    ) -> PeftModel:
        # PEFT can mutate the config. Keep the caller's config reusable.
        config = deepcopy(config)

        if self._custom_modules:
            config._register_custom_module(self._custom_modules)

        with PeftLoraInjector.disable_peft_config_converting_for_difused_experts(model):
            return cast(PeftModel, get_peft_model(
                model, config, adapter_name=adapter_name,
                autocast_adapter_dtype=autocast_adapter_dtype,
            ))

    @contextmanager
    @staticmethod
    def disable_peft_config_converting_for_difused_experts(
        base_model: PreTrainedModel,
    ) -> Generator[None, None, None]:
        """Temporarily skip PEFT's target conversion for a model with separate experts."""
        has_separate_experts = is_transformers_ge_v5 and any(
            name.endswith(".experts") and any(key.isdecimal() for key, _ in module.named_children())
            for name, module in base_model.named_modules()
        )
        if not has_separate_experts:
            yield
            return

        from peft.utils import transformers_weight_conversion

        with _expert_conversion_lock:
            convert_config = transformers_weight_conversion.convert_peft_config_for_transformers

            def convert_for_other_models(
                peft_config: PeftConfig, model: nn.Module, conversions: list[Any] | None,
            ) -> None:
                if model is not base_model:
                    convert_config(peft_config, model, conversions)

            with patch.object(
                transformers_weight_conversion, "convert_peft_config_for_transformers", convert_for_other_models,
            ):
                yield
