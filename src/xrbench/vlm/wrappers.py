"""Static-shape, tensor-only VLM stage wrappers."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # Allows light CLI/tests to import the package without the VLM extra.
    torch = None  # type: ignore[assignment]

    class _NN:
        Module = object

    nn = _NN()  # type: ignore[assignment]

from xrbench.vlm.cache_adapter import dynamic_cache_from_flat, flatten_kv_cache


def _first_tensor(output: Any) -> Any:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple | list):
        return output[0]
    return output


class VisionEncoderWrapper(nn.Module):  # type: ignore[misc]
    def __init__(self, vision_module: Any) -> None:
        super().__init__()
        self.vision_module = vision_module

    def forward(self, pixel_values: Any, patch_attention_mask: Any) -> Any:
        output = self.vision_module(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask.to(dtype=torch.bool),
            return_dict=False,
        )
        return _first_tensor(output)


class VisionProjectorWrapper(nn.Module):  # type: ignore[misc]
    def __init__(self, connector_module: Any) -> None:
        super().__init__()
        self.connector_module = connector_module

    def forward(self, visual_features: Any) -> Any:
        return _first_tensor(self.connector_module(visual_features))


class LanguagePrefillWrapper(nn.Module):  # type: ignore[misc]
    """Return final-position logits followed by flattened KV tensors."""

    def __init__(self, language_module: Any, lm_head: Any) -> None:
        super().__init__()
        self.language_module = language_module
        self.lm_head = lm_head

    def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, ...]:
        output = self.language_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        hidden = _first_tensor(output)
        logits = self.lm_head(hidden[:, -1:, :])
        return (logits, *flatten_kv_cache(output.past_key_values))


class LanguageDecodeWrapper(nn.Module):  # type: ignore[misc]
    """Consume one token and explicit flattened fixed-context KV tensors."""

    def __init__(self, language_module: Any, lm_head: Any) -> None:
        super().__init__()
        self.language_module = language_module
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        cache_position: Any,
        *past_key_values: Any,
    ) -> tuple[Any, ...]:
        cache = dynamic_cache_from_flat(past_key_values)
        output = self.language_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        hidden = _first_tensor(output)
        logits = self.lm_head(hidden[:, -1:, :])
        return (logits, *flatten_kv_cache(output.past_key_values))
