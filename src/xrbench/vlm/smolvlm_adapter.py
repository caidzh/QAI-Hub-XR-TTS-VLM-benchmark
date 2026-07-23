"""Configurable Hugging Face SmolVLM loading and deterministic sample construction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrbench.errors import ConfigurationError, ExportError
from xrbench.vlm.architecture_inspector import (
    ArchitectureReport,
    build_report,
    get_module,
    resolve_architecture,
)
from xrbench.vlm.wrappers import (
    LanguageDecodeWrapper,
    LanguagePrefillWrapper,
    VisionEncoderWrapper,
    VisionProjectorWrapper,
)


@dataclass(frozen=True)
class VisionSample:
    pixel_values: Any
    patch_attention_mask: Any
    visual_features: Any
    projected_features: Any


class SmolVLMAdapter:
    def __init__(self, model_id: str, model: Any, processor: Any) -> None:
        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.vision_info, self.connector_info, self.language_info = resolve_architecture(model)
        self.vision_module = get_module(model, self.vision_info.path)
        self.connector_module = get_module(model, self.connector_info.path)
        self.language_module = get_module(model, self.language_info.path)
        self.lm_head = _resolve_lm_head(model)

    @classmethod
    def from_pretrained(cls, model_id: str) -> SmolVLMAdapter:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            torch.manual_seed(0)
            random.seed(0)
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
        except Exception as error:
            raise ConfigurationError(
                f"Could not load VLM '{model_id}'. Install the `vlm` extra and ensure "
                "the model is accessible through Hugging Face."
            ) from error
        return cls(model_id, model, processor)

    def vision_sample(self, image_path: Path) -> VisionSample:
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise ConfigurationError("Pillow and torch are required for VLM preparation") from error
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            processed = self.processor(images=image_rgb, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(torch.float32)
        # SmolVLM processors expose [B, num_images, C, H, W]; the isolated
        # vision module consumes [B*num_images, C, H, W].
        if pixel_values.ndim == 5:
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
        patch_size = int(getattr(self.vision_module.config, "patch_size", 1))
        patch_attention_mask = torch.ones(
            (
                pixel_values.shape[0],
                pixel_values.shape[-2] // patch_size,
                pixel_values.shape[-1] // patch_size,
            ),
            dtype=torch.int32,
        )
        vision_wrapper = VisionEncoderWrapper(self.vision_module).eval()
        projector_wrapper = VisionProjectorWrapper(self.connector_module).eval()
        with torch.inference_mode():
            visual = vision_wrapper(pixel_values, patch_attention_mask)
            projected = projector_wrapper(visual)
        return VisionSample(pixel_values, patch_attention_mask, visual, projected)

    def architecture_report(self, sample: VisionSample) -> ArchitectureReport:
        return build_report(
            self.model,
            sample.pixel_values,
            sample.visual_features,
            sample.projected_features,
        )

    def wrappers(self) -> dict[str, Any]:
        return {
            "vision_encoder": VisionEncoderWrapper(self.vision_module).eval(),
            "vision_projector": VisionProjectorWrapper(self.connector_module).eval(),
            "language_prefill": LanguagePrefillWrapper(
                self.language_module, self.lm_head
            ).eval(),
            "language_decode": LanguageDecodeWrapper(
                self.language_module, self.lm_head
            ).eval(),
        }

    def token_inputs(self, prompt: str, sequence_length: int) -> tuple[Any, Any]:
        import torch

        encoded = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=sequence_length,
        )
        input_ids = encoded["input_ids"].to(torch.int64)
        attention_mask = encoded["attention_mask"].to(torch.int64)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id or 0
        if input_ids.shape[1] < sequence_length:
            padding = sequence_length - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(input_ids, (0, padding), value=int(pad_id))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding), value=0)
        if input_ids.shape[1] != sequence_length:
            raise ExportError(
                f"Tokenizer could not produce fixed sequence length {sequence_length}: "
                f"{tuple(input_ids.shape)}"
            )
        return input_ids, attention_mask

    def empty_cache(self, context_length: int) -> tuple[Any, ...]:
        import torch

        config = self.language_module.config
        layers = int(config.num_hidden_layers)
        heads = int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)
        )
        head_dim = int(
            getattr(
                config,
                "head_dim",
                int(config.hidden_size) // int(config.num_attention_heads),
            )
        )
        dtype = next(self.language_module.parameters()).dtype
        tensors: list[Any] = []
        for _ in range(layers):
            tensors.extend(
                (
                    torch.zeros((1, heads, context_length, head_dim), dtype=dtype),
                    torch.zeros((1, heads, context_length, head_dim), dtype=dtype),
                )
            )
        return tuple(tensors)


def _resolve_lm_head(model: Any) -> Any:
    direct = getattr(model, "lm_head", None)
    if direct is not None:
        return direct
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        output = getter()
        if output is not None:
            return output
    candidates = [
        (path, module)
        for path, module in model.named_modules()
        if path and any(token in path.lower() for token in ("lm_head", "output_projection"))
    ]
    if candidates:
        return sorted(candidates, key=lambda item: (len(item[0]), item[0]))[0][1]
    raise ExportError("Could not resolve the language-model output head")
