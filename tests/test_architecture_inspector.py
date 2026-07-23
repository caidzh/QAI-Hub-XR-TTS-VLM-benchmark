from __future__ import annotations

from types import SimpleNamespace

from xrbench.vlm.architecture_inspector import resolve_architecture


class Vision:
    config = SimpleNamespace(image_size=224)


class Connector:
    pass


class Projection:
    pass


class Language:
    config = SimpleNamespace(num_hidden_layers=4)

    def get_input_embeddings(self) -> None:
        return None


class Model:
    def named_modules(self):
        return [
            ("", self),
            ("model.vision_model", Vision()),
            ("model.vision_model.encoder.layers.0", object()),
            ("model.connector", Connector()),
            ("model.connector.modality_projection", Projection()),
            ("model.text_model", Language()),
            ("model.text_model.layers.0", object()),
        ]


def test_architecture_resolution_prefers_complete_shallow_stages() -> None:
    vision, connector, language = resolve_architecture(Model())
    assert vision.path == "model.vision_model"
    assert connector.path == "model.connector"
    assert language.path == "model.text_model"
