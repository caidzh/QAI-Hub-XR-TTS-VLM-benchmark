"""Runtime architecture discovery without one fragile hardcoded module path."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xrbench.errors import ExportError


@dataclass(frozen=True)
class ResolvedModule:
    path: str
    class_name: str


@dataclass(frozen=True)
class ArchitectureReport:
    model_class: str
    vision_module: ResolvedModule
    connector_module: ResolvedModule
    language_module: ResolvedModule
    image_tensor_shape: tuple[int, ...]
    image_tensor_dtype: str
    visual_feature_shape: tuple[int, ...]
    projected_visual_token_shape: tuple[int, ...]
    vocabulary_size: int | None
    number_of_layers: int | None
    number_of_attention_heads: int | None
    number_of_key_value_heads: int | None
    head_dimension: int | None
    kv_cache_tensor_organization: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def qualified_class(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _named_modules(model: Any) -> list[tuple[str, Any]]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        raise ExportError(f"Model {qualified_class(model)} does not expose named_modules()")
    return [(str(path), module) for path, module in method()]


def _ranked_match(
    modules: Iterable[tuple[str, Any]],
    *,
    path_tokens: tuple[str, ...],
    class_tokens: tuple[str, ...],
    predicate: Any | None = None,
    exclude: set[str] | None = None,
) -> tuple[str, Any]:
    excluded = exclude or set()
    candidates: list[tuple[int, int, str, Any]] = []
    for path, module in modules:
        if not path or path in excluded:
            continue
        path_lower = path.lower()
        leaf = path_lower.rsplit(".", 1)[-1]
        class_lower = module.__class__.__name__.lower()
        path_score = max(
            (30 if token == leaf else 20 for token in path_tokens if token in path_lower),
            default=0,
        )
        class_score = max(
            (10 for token in class_tokens if token in class_lower),
            default=0,
        )
        score = path_score + class_score
        if predicate is not None and predicate(module):
            score += 8
        if score:
            candidates.append((-score, len(path.split(".")), path, module))
    if not candidates:
        raise ExportError(
            f"Could not resolve module using path tokens {path_tokens} and class tokens {class_tokens}"
        )
    _, _, path, module = sorted(candidates, key=lambda value: (value[0], value[1], value[2]))[0]
    return path, module


def resolve_architecture(model: Any) -> tuple[ResolvedModule, ResolvedModule, ResolvedModule]:
    modules = _named_modules(model)
    vision_path, vision = _ranked_match(
        modules,
        path_tokens=("vision_model", "vision_tower", "vision_encoder"),
        class_tokens=("visiontransformer", "visionmodel", "visionencoder"),
        predicate=lambda item: hasattr(item, "config")
        and hasattr(getattr(item, "config", None), "image_size"),
    )
    connector_path, connector = _ranked_match(
        modules,
        path_tokens=("connector", "projector", "multi_modal_projector", "modality_projection"),
        class_tokens=("connector", "projector", "projection"),
        exclude={vision_path},
    )
    language_path, language = _ranked_match(
        modules,
        path_tokens=("text_model", "language_model", "language"),
        class_tokens=("causallm", "textmodel", "llamamodel"),
        predicate=lambda item: callable(getattr(item, "get_input_embeddings", None))
        and hasattr(item, "config"),
        exclude={vision_path, connector_path},
    )
    return (
        ResolvedModule(vision_path, qualified_class(vision)),
        ResolvedModule(connector_path, qualified_class(connector)),
        ResolvedModule(language_path, qualified_class(language)),
    )


def get_module(model: Any, path: str) -> Any:
    current = model
    for part in path.split("."):
        current = getattr(current, part)
    return current


def build_report(
    model: Any,
    image_tensor: Any,
    visual_features: Any,
    projected_features: Any,
) -> ArchitectureReport:
    vision, connector, language = resolve_architecture(model)
    language_module = get_module(model, language.path)
    config = getattr(language_module, "config", getattr(model, "config", None))
    layers = _config_int(config, "num_hidden_layers", "n_layer", "num_layers")
    heads = _config_int(config, "num_attention_heads", "n_head")
    kv_heads = _config_int(config, "num_key_value_heads") or heads
    hidden = _config_int(config, "hidden_size", "n_embd")
    head_dim = _config_int(config, "head_dim")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    cache = (
        f"flattened alternating key/value tensors: 2 tensors/layer, each "
        f"[batch, {kv_heads or 'kv_heads'}, fixed_context, {head_dim or 'head_dim'}]; "
        "inputs and outputs are ordered key_0,value_0,key_1,value_1,..."
    )
    return ArchitectureReport(
        model_class=qualified_class(model),
        vision_module=vision,
        connector_module=connector,
        language_module=language,
        image_tensor_shape=tuple(map(int, image_tensor.shape)),
        image_tensor_dtype=str(image_tensor.dtype),
        visual_feature_shape=tuple(map(int, visual_features.shape)),
        projected_visual_token_shape=tuple(map(int, projected_features.shape)),
        vocabulary_size=_config_int(config, "vocab_size"),
        number_of_layers=layers,
        number_of_attention_heads=heads,
        number_of_key_value_heads=kv_heads,
        head_dimension=head_dim,
        kv_cache_tensor_organization=cache,
    )


def _config_int(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def write_report(report: ArchitectureReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "architecture.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# VLM architecture diagnostic",
        "",
        f"- Full model class: `{report.model_class}`",
        f"- Vision module: `{report.vision_module.class_name}` at `{report.vision_module.path}`",
        f"- Connector/projector: `{report.connector_module.class_name}` at `{report.connector_module.path}`",
        f"- Language model: `{report.language_module.class_name}` at `{report.language_module.path}`",
        f"- Image tensor: `{report.image_tensor_shape}` / `{report.image_tensor_dtype}`",
        f"- Visual features: `{report.visual_feature_shape}`",
        f"- Projected visual tokens: `{report.projected_visual_token_shape}`",
        f"- Vocabulary size: `{report.vocabulary_size}`",
        f"- Layers / attention heads / KV heads: `{report.number_of_layers}` / "
        f"`{report.number_of_attention_heads}` / `{report.number_of_key_value_heads}`",
        f"- KV cache: {report.kv_cache_tensor_organization}",
        "",
    ]
    (output_dir / "architecture.md").write_text("\n".join(lines), encoding="utf-8")
