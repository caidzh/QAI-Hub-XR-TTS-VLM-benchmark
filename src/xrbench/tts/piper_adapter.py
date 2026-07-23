"""Lazy adapter around the official Qualcomm AI Hub Models PiperTTS package."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from xrbench.errors import ConfigurationError


@dataclass(frozen=True)
class PreparedComponent:
    name: str
    source_path: Path
    source_format: str
    input_specs: dict[str, tuple[tuple[int, ...], str]]
    compile_options: str
    link_options: str
    profile_options: str


def _tensor_spec(spec: Any) -> tuple[tuple[int, ...], str]:
    shape = getattr(spec, "shape", None)
    dtype = getattr(spec, "dtype", None)
    if shape is None and isinstance(spec, tuple):
        if len(spec) == 2 and isinstance(spec[0], tuple):
            return tuple(map(int, spec[0])), str(spec[1])
        return tuple(map(int, spec)), "float32"
    if shape is None:
        raise ConfigurationError(f"Unsupported Qualcomm input spec: {spec!r}")
    return tuple(map(int, shape)), str(dtype or "float32")


class PiperAdapter:
    """Load, run, and serialize only through the official model package."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def from_pretrained(cls, *, synthesis_only: bool = False) -> PiperAdapter:
        try:
            import torch

            torch.manual_seed(0)
            module = importlib.import_module("qai_hub_models.models.pipertts_en")
            if synthesis_only:
                shared = importlib.import_module(
                    "qai_hub_models.models._shared.pipertts.model"
                )
                language = module.Model.get_language()
                generator = shared.get_model(language)
                model = SimpleNamespace(
                    encoder=shared.Encoder(generator),
                    sdp=shared.SDP(generator, shared.SPEED[language]),
                    flow=shared.Flow(generator),
                    decoder=shared.Decoder(generator),
                    get_language=module.Model.get_language,
                )
            else:
                model = module.Model.from_pretrained()
        except Exception as error:
            raise ConfigurationError(
                "Official PiperTTS-EN could not be loaded. Install the `tts` extra and "
                "the upstream Piper Python package with scripts/bootstrap.sh --tts."
            ) from error
        return cls(model)

    @property
    def component_names(self) -> list[str]:
        return list(self.model.component_names)

    def prepare_components(self, output_dir: Path) -> list[PreparedComponent]:
        try:
            from qai_hub_models import Precision, TargetRuntime
        except ImportError as error:
            raise ConfigurationError("qai-hub-models is required for TTS preparation") from error
        output_dir.mkdir(parents=True, exist_ok=True)
        input_specs = self.model.get_input_spec()
        prepared: list[PreparedComponent] = []
        for name in self.component_names:
            source = Path(
                self.model.serialize_component(name, output_dir, input_specs.get(name))
            ).resolve()
            options = self.model.get_component_hub_compile_options(
                name,
                TargetRuntime.VOICE_AI,
                Precision.float,
            )
            link_options = self.model.get_component_hub_link_options(
                name, TargetRuntime.VOICE_AI
            )
            profile_options = self.model.get_component_hub_profile_options(
                name, TargetRuntime.VOICE_AI
            )
            specs = {
                input_name: _tensor_spec(value)
                for input_name, value in input_specs[name].items()
            }
            prepared.append(
                PreparedComponent(
                    name=name,
                    source_path=source,
                    source_format=_source_format(source),
                    input_specs=specs,
                    compile_options=str(options),
                    link_options=str(link_options),
                    profile_options=str(profile_options),
                )
            )
        metadata_path = output_dir / "components.json"
        metadata_path.write_text(
            json.dumps(
                [
                    {
                        "name": item.name,
                        "source_path": str(item.source_path),
                        "source_format": item.source_format,
                        "input_specs": {
                            key: {"shape": value[0], "dtype": value[1]}
                            for key, value in item.input_specs.items()
                        },
                        "compile_options": item.compile_options,
                        "link_options": item.link_options,
                        "profile_options": item.profile_options,
                    }
                    for item in prepared
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return prepared

    def sample_inputs(self, component: str) -> Mapping[str, list[Any]]:
        samples = self.model.get_component_sample_inputs(
            component, self.model.get_component_input_spec(component), False
        )
        return {
            str(name): [
                value.detach().cpu().numpy() if hasattr(value, "detach") else value
                for value in values
            ]
            for name, values in samples.items()
        }

    def reference_outputs(
        self, component: str, inputs: Mapping[str, list[Any]]
    ) -> dict[str, Any]:
        import torch

        module = self.model.components[component]
        ordered = [
            torch.as_tensor(values[0]) if not hasattr(values[0], "detach") else values[0]
            for values in inputs.values()
        ]
        with torch.inference_mode():
            output = module(*ordered)
        values = tuple(output) if isinstance(output, tuple | list) else (output,)
        names = list(self.model.get_component_output_spec(component))
        return {
            names[index] if index < len(names) else f"output_{index}": (
                value.detach().cpu().numpy() if hasattr(value, "detach") else value
            )
            for index, value in enumerate(values)
        }


def _source_format(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".onnx": "onnx",
        ".pt2": "pt2",
        ".pt": "torchscript",
        ".pth": "torchscript",
    }.get(suffix, suffix.lstrip(".") or "unknown")
