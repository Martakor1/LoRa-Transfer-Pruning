x → convert₁ → SP-маска → revert₁ → x_masked → следующие слои → loss
                                       │
                                       └→ convert₂ → capture → конец

live тензоры в capture хоть и сохраняются, но не участвуют в dep graph, и когда
в cache_activation_and_gradient вызывается loss(live[name]) то в gradient[name] записывается None

Смотри test_pruning_conversion_backward_requires_upstream_anchor в LoRa-Transfer-Pruning/tests/pruning_lens/adapter/torch_pruning/test_torch_pruning_tracer.py
