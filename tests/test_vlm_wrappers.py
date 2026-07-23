from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from xrbench.vlm.cache_adapter import flatten_kv_cache, pairs_from_flat  # noqa: E402
from xrbench.vlm.wrappers import VisionEncoderWrapper, VisionProjectorWrapper  # noqa: E402


class Vision(torch.nn.Module):
    def forward(self, pixel_values, patch_attention_mask, return_dict=False):
        output = pixel_values.flatten(2).transpose(1, 2)
        return (output,) if not return_dict else type("Output", (), {"last_hidden_state": output})


class Projector(torch.nn.Module):
    def forward(self, value):
        return value * 2


def test_vision_and_projector_wrapper_parity() -> None:
    pixels = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    mask = torch.ones((1, 1, 1), dtype=torch.bool)
    vision = Vision()
    direct = vision(pixels, mask)[0]
    wrapped = VisionEncoderWrapper(vision)(pixels, mask)
    assert torch.equal(direct, wrapped)
    assert torch.equal(VisionProjectorWrapper(Projector())(wrapped), direct * 2)


def test_cache_flattening_order() -> None:
    cache = ((torch.tensor([1]), torch.tensor([2])), (torch.tensor([3]), torch.tensor([4])))
    flat = flatten_kv_cache(cache)
    assert [int(value) for tensor in flat for value in tensor] == [1, 2, 3, 4]
    assert len(pairs_from_flat(flat)) == 2
