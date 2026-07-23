"""Instrumented local floating-point PiperTTS reference pipeline."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xrbench.errors import ConfigurationError
from xrbench.tts.piper_adapter import PiperAdapter


@dataclass(frozen=True)
class LocalTTSResult:
    variant: str
    text: str
    normalized_text: str
    raw_character_count: int
    normalized_text_length: int
    phoneme_count: int
    actual_model_input_sequence_length: int
    generated_sample_count: int
    generated_audio_duration_seconds: float
    sample_rate: int
    invocation_counts: dict[str, int]
    timing_ms: dict[str, float]
    wav_path: str


def normalize_text(text: str) -> str:
    """Apply conservative whitespace normalization without changing punctuation."""

    return re.sub(r"\s+", " ", text).strip()


def synthesize_reference(
    adapter: PiperAdapter,
    text: str,
    variant: str,
    output_dir: Path,
) -> LocalTTSResult:
    """Run the official component implementations while timing each pipeline domain."""

    try:
        import soundfile as sf
        import torch
        from qai_hub_models.models._shared.pipertts.app import (
            LANGUAGE_MAP_ph,
            noise_scale_for_language,
            phonemize_text,
        )
        from qai_hub_models.models._shared.pipertts.model import (
            DEC_SEQ_OVERLAP,
            DEFAULT_LENGTH_SCALE,
            DEFAULT_NOISE_SCALE_W,
            MAX_DEC_SEQ_LEN,
            MAX_SEQ_LEN,
            SAMPLE_RATE,
            UPSAMPLE_FACTOR,
            UPSAMPLED_MAX_SEQ_LEN,
        )
        from qai_hub_models.models._shared.voiceai_tts.app_utils import generate_path
    except ImportError as error:
        raise ConfigurationError(
            "The official PiperTTS runtime is incomplete; run scripts/bootstrap.sh --tts"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    invocation_counts = {
        "encoder": 0,
        "sdp": 0,
        "flow": 0,
        "decoder": 0,
        "charsiu_encoder": 0,
        "charsiu_decoder": 0,
        "autoregressive_or_iterative": 0,
    }
    timings: dict[str, float] = {}
    total_start = time.perf_counter_ns()

    started = time.perf_counter_ns()
    normalized = normalize_text(text)
    timings["text_normalization"] = _elapsed_ms(started)

    started = time.perf_counter_ns()
    language = adapter.model.get_language()
    phoneme_ids = phonemize_text(normalized, LANGUAGE_MAP_ph[language])
    timings["phonemization"] = _elapsed_ms(started)
    if len(phoneme_ids) > MAX_SEQ_LEN:
        raise ConfigurationError(
            f"TTS input '{variant}' produced {len(phoneme_ids)} phoneme IDs, exceeding "
            f"the official model maximum of {MAX_SEQ_LEN}; refusing implicit truncation"
        )

    started = time.perf_counter_ns()
    padded = phoneme_ids + [0] * (MAX_SEQ_LEN - len(phoneme_ids))
    x = torch.tensor([padded], dtype=torch.int32)
    x_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.int32)
    length_scale = torch.tensor([DEFAULT_LENGTH_SCALE], dtype=torch.float32)
    noise_scale_w = torch.tensor([DEFAULT_NOISE_SCALE_W], dtype=torch.float32)
    noise_scale = torch.tensor([noise_scale_for_language(language)], dtype=torch.float32)
    timings["tensor_preparation"] = _elapsed_ms(started)

    with torch.inference_mode():
        started = time.perf_counter_ns()
        x_encoded, m_p, logs_p, x_mask = adapter.model.encoder(x, x_lengths)
        timings["encoder_neural"] = _elapsed_ms(started)
        invocation_counts["encoder"] += 1

        started = time.perf_counter_ns()
        y_lengths, w_ceil = adapter.model.sdp(
            x_encoded, x_mask, length_scale, noise_scale_w
        )
        timings["sdp_neural"] = _elapsed_ms(started)
        invocation_counts["sdp"] += 1

        started = time.perf_counter_ns()
        y_mask = torch.unsqueeze(
            torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1), dim=1
        ).to(torch.float32)
        attention_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
        attention = generate_path(w_ceil, attention_mask)
        attention_squeezed = attention.squeeze(1).to(torch.float32)
        timings["alignment_preparation"] = _elapsed_ms(started)

        started = time.perf_counter_ns()
        z = adapter.model.flow(
            m_p.to(torch.float32),
            logs_p.to(torch.float32),
            y_mask,
            attention_squeezed,
            noise_scale,
        )
        timings["flow_neural"] = _elapsed_ms(started)
        invocation_counts["flow"] += 1

        audio, decoder_ms, decoder_calls = _decode_instrumented(
            adapter.model.decoder,
            z,
            y_lengths,
            max_dec_seq_len=MAX_DEC_SEQ_LEN,
            overlap=DEC_SEQ_OVERLAP,
            upsample_factor=UPSAMPLE_FACTOR,
        )
        timings["decoder_neural"] = decoder_ms
        invocation_counts["decoder"] = decoder_calls
        invocation_counts["autoregressive_or_iterative"] = max(decoder_calls - 1, 0)

    generated_samples = min(int(y_lengths[0]) * UPSAMPLE_FACTOR, int(audio.numel()))
    audio_array = audio.reshape(-1)[:generated_samples].detach().cpu().numpy()
    wav_path = output_dir / f"{variant}.wav"
    started = time.perf_counter_ns()
    sf.write(wav_path, audio_array, SAMPLE_RATE)
    timings["wav_serialization"] = _elapsed_ms(started)
    timings["local_float_neural_inference"] = sum(
        timings[key]
        for key in ("encoder_neural", "sdp_neural", "flow_neural", "decoder_neural")
    )
    timings["local_total_synthesis"] = (time.perf_counter_ns() - total_start) / 1_000_000.0

    return LocalTTSResult(
        variant=variant,
        text=text,
        normalized_text=normalized,
        raw_character_count=len(text),
        normalized_text_length=len(normalized),
        phoneme_count=len(phoneme_ids),
        actual_model_input_sequence_length=len(phoneme_ids),
        generated_sample_count=generated_samples,
        generated_audio_duration_seconds=generated_samples / float(SAMPLE_RATE),
        sample_rate=SAMPLE_RATE,
        invocation_counts=invocation_counts,
        timing_ms=timings,
        wav_path=str(wav_path.resolve()),
    )


def _decode_instrumented(
    decoder: Any,
    z: Any,
    y_lengths: Any,
    *,
    max_dec_seq_len: int,
    overlap: int,
    upsample_factor: int,
) -> tuple[Any, float, int]:
    import torch

    z_buffer = torch.zeros(
        [z.shape[0], z.shape[1], max_dec_seq_len + 2 * overlap],
        dtype=torch.float32,
    )
    z_buffer[:, :, : max_dec_seq_len + overlap] = z[:, :, : max_dec_seq_len + overlap]
    started = time.perf_counter_ns()
    audio_chunk = decoder(z_buffer)
    elapsed = _elapsed_ms(started)
    calls = 1
    audio = audio_chunk.squeeze()[: max_dec_seq_len * upsample_factor]
    total_decoded = max_dec_seq_len
    while total_decoded < min(
        int(y_lengths[0]), int(z.shape[2]) - max_dec_seq_len - overlap
    ):
        z_buffer = z[
            :,
            :,
            total_decoded - overlap : total_decoded + max_dec_seq_len + overlap,
        ]
        started = time.perf_counter_ns()
        audio_chunk = decoder(z_buffer)
        elapsed += _elapsed_ms(started)
        calls += 1
        audio_chunk = audio_chunk.squeeze()[
            overlap * upsample_factor : (max_dec_seq_len + overlap) * upsample_factor
        ]
        audio = torch.cat([audio, audio_chunk])
        total_decoded += max_dec_seq_len
    return audio, elapsed, calls


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def load_sentences(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ConfigurationError(f"TTS sentences file must map names to strings: {path}")
    return data


def run_local_suite(
    sentences: Mapping[str, str],
    output_dir: Path,
    *,
    adapter: PiperAdapter | None = None,
) -> list[LocalTTSResult]:
    adapter = adapter or PiperAdapter.from_pretrained()
    results = [
        synthesize_reference(adapter, text, variant, output_dir)
        for variant, text in sentences.items()
    ]
    (output_dir / "local_metrics.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
