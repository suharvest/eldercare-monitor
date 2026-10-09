"""App-local single-context T172 SenseVoice backend.

The production path intentionally reuses the parent RK backend's broker,
remote-session, lifecycle, drain, READY, and quarantine behavior.  Only the
encoder shape and long-audio stitching differ from the ordinary T344/T100
metadata path.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from kit.errors import ConfigurationError
from kit.runtime.remote import configured_inference_socket
from kit.asr_rknn_backend import RknnSenseVoiceBackend
from voxedge.backends.base import TranscriptionResult


PREFIX_FRAMES = 4
AUDIO_SPAN = 168
WINDOW_T = PREFIX_FRAMES + AUDIO_SPAN
OVERLAP = 32
STRIDE = AUDIO_SPAN - OVERLAP
CENTER = OVERLAP // 2
BLANK = 0
MODEL_BYTES = 246399355
MODEL_SHA256 = "4f73132adf3235ed32152617df16c69d8581db5f1bf1d71278302bf6e2d015fb"

# Installed-platform receipt, also verified against the checked-out source.
# Production startup stays closed if either module resolves to another source.
PARENT_SDK_SHA256 = {
    "kit.asr": "a3d392c15bcc254ad2badc6a6390d591ce249a3129a343f4c60b11d8da7b77d7",
    "kit.asr_rknn_backend": "f84b946924d49db6589d514809f2386c0c8a9c99a086c35ccc33ebd14596e8e4",
}
BPE_BYTES = 377341
BPE_SHA256 = "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
BPE_PIECE_SIZE = 25055
MODEL_PROFILE = "t172_scaled_hybrid"
_DECODE_METRICS = threading.local()


def get_last_decode_metrics() -> dict:
    """Return the current thread's last complete decode metrics."""
    return dict(getattr(_DECODE_METRICS, "value", {}))


def _window_plan(n_audio: int) -> list[dict]:
    if n_audio < 0:
        raise ValueError("negative audio frame count")
    if n_audio == 0:
        return []
    starts = [0]
    while starts[-1] + AUDIO_SPAN < n_audio:
        starts.append(starts[-1] + STRIDE)
    out = []
    for index, start in enumerate(starts):
        actual = min(AUDIO_SPAN, n_audio - start)
        emit_start = 0 if index == 0 else start + CENTER
        emit_end = (n_audio if index + 1 == len(starts)
                    else starts[index + 1] + CENTER)
        if not (start <= emit_start < emit_end <= start + actual):
            raise AssertionError((n_audio, start, actual, emit_start, emit_end))
        out.append({"start": start, "end": start + actual,
                    "actual": actual, "pad": AUDIO_SPAN - actual,
                    "emit_start": emit_start, "emit_end": emit_end,
                    "emit_start_local": emit_start - start,
                    "emit_end_local": emit_end - start})
    return out


def _module_source_path(module_name: str) -> str:
    module = importlib.import_module(module_name)
    source = inspect.getsourcefile(module)
    if not source:
        raise ConfigurationError(
            f"cannot resolve source for parent SDK module {module_name}",
            operation="eldercare.asr_t172", code="sdk_source_unavailable")
    return str(Path(source).resolve())


def _verify_parent_sdk(checker=None) -> None:
    for module_name, expected in PARENT_SDK_SHA256.items():
        source = _module_source_path(module_name)
        if expected is None:
            raise ConfigurationError(
                f"parent SDK source identity pending device receipt: {module_name}",
                operation="eldercare.asr_t172", code="sdk_identity_pending")
        actual = (checker(source) if checker is not None else
                  hashlib.sha256(Path(source).read_bytes()).hexdigest())
        if expected is not None and actual != expected:
            raise ConfigurationError(
                f"parent SDK source hash mismatch: {module_name}",
                operation="eldercare.asr_t172", code="sdk_identity_mismatch")


def _verify_model(path: str) -> None:
    size = 0
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    actual = digest.hexdigest()
    if (size, actual) != (MODEL_BYTES, MODEL_SHA256):
        raise ConfigurationError(
            f"unexpected T172 model bytes/hash: {size} {actual}",
            operation="eldercare.asr_t172", code="model_identity_mismatch")


def _verify_bpe(path: str) -> None:
    size = 0
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    actual = digest.hexdigest()
    if (size, actual) != (BPE_BYTES, BPE_SHA256):
        raise ConfigurationError(
            f"unexpected BPE bytes/hash: {size} {actual}",
            operation="eldercare.asr_t172", code="bpe_identity_mismatch")


def _verify_bpe_piece_size(path: str) -> None:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    if sp.load(path) is False:
        raise ConfigurationError(
            f"sentencepiece load failed: {path}",
            operation="eldercare.asr_t172", code="bpe_load_failed")
    piece_size = int(sp.get_piece_size())
    if piece_size != BPE_PIECE_SIZE:
        raise ConfigurationError(
            f"unexpected BPE piece size: {piece_size}",
            operation="eldercare.asr_t172", code="bpe_piece_size_mismatch")


def _detokenize(sp, ids: list[int]) -> tuple[str, str]:
    raw = "".join(sp.id_to_piece(i) for i in ids
                   if 0 <= i < sp.get_piece_size()).replace("▁", " ")
    match = re.search(r"<\|(zh|en|yue|ja|ko|nospeech)\|>", raw)
    detected = match.group(1) if match else ""
    return re.sub(r"<\|[^|]*\|>", "", raw), detected


class T172SenseVoiceBackend(RknnSenseVoiceBackend):
    """Single T172 context with exact center32 long-audio stitching."""

    def __init__(self, model: str, cmvn: str, embedding: str, bpe: str,
                 *, language: str = "auto", debug: bool = False,
                 lease=None, lease_factory=None, runtime_factory=None,
                 sentencepiece_factory=None, sdk_hash_checker=None):
        if any(value is not None for value in
               (lease, lease_factory, runtime_factory, sentencepiece_factory)):
            raise ConfigurationError(
                "T172 production backend forbids local lease/runtime injection",
                operation="eldercare.asr_t172", code="production_injection_forbidden")
        service = configured_inference_socket()
        if not service:
            raise ConfigurationError(
                "T172 production backend requires configured managed inference",
                operation="eldercare.asr_t172", code="managed_inference_required")
        _verify_parent_sdk(sdk_hash_checker)
        _verify_model(model)
        # Verify tokenizer bytes and piece count before constructing the
        # parent backend, so a bad tokenizer cannot trigger any NPU/session
        # initialization.  The parent only loads its own SentencePiece model
        # inside preload(), so self._sp is still None after __init__.
        _verify_bpe(bpe)
        _verify_bpe_piece_size(bpe)
        super().__init__(
            model, cmvn, embedding, bpe, rknn_model_short=None,
            language=language, textnorm="woitn", debug=debug,
            lease=lease, lease_factory=lease_factory,
            runtime_factory=runtime_factory,
            sentencepiece_factory=sentencepiece_factory)

    def _load_one(self, path: str, *, frames: int):
        runtime = super()._load_one(path, frames=WINDOW_T)
        shape = getattr(getattr(runtime, "spec", None), "inputs", ())
        if shape:
            declared = getattr(shape[0], "shape", None)
            if declared and WINDOW_T not in tuple(declared):
                raise ConfigurationError(
                    f"T172 runtime metadata shape mismatch: {declared}",
                    operation="eldercare.asr_t172", code="shape_mismatch")
        return runtime

    def _transcribe_array_impl(self, samples: np.ndarray, language: str = "auto"):
        started = time.monotonic()
        lang = self._resolve_lang(language)
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        if audio.size == 0:
            _DECODE_METRICS.value = {
                "model_profile": MODEL_PROFILE,
                "model_sha256": MODEL_SHA256,
                "window_t": WINDOW_T, "audio_span": AUDIO_SPAN,
                "overlap": OVERLAP, "stitch_policy": "t172_center32",
                "frames_in": 0, "frames_covered": 0,
                "finite_full": True, "output_shapes": [],
                "elapsed_sec": time.monotonic() - started,
            }
            return TranscriptionResult(text="", language=lang if lang != "auto" else "")
        sp_in = self._prep(audio, lang)
        prefix, lfr = sp_in[:PREFIX_FRAMES], sp_in[PREFIX_FRAMES:]
        plan = _window_plan(int(lfr.shape[0]))
        if not plan and lfr.shape[0]:
            raise AssertionError("nonempty audio produced no T172 window")
        pieces = []
        detected_language = ""
        finite_full = True
        output_shapes = []
        for window in plan:
            self._ensure_lease_alive()
            chunk = lfr[window["start"]:window["end"]]
            if window["actual"] < AUDIO_SPAN:
                chunk = np.vstack([chunk, np.zeros(
                    (window["pad"], lfr.shape[1]), dtype=np.float32)])
            inp = np.concatenate([prefix, chunk], axis=0)
            if inp.shape != (WINDOW_T, 560):
                raise AssertionError(inp.shape)
            if self._inference_service:
                outputs = self._rknn.infer(inp[None].astype(np.float32))
            else:
                outputs = self._rknn.inference(
                    inputs=[inp[None].astype(np.float32)])
            if len(outputs) != 1:
                raise RuntimeError(f"T172 expected one RKNN output, got {len(outputs)}")
            output = np.asarray(outputs[0])
            vocab = int(self._sp.get_piece_size())
            if output.shape != (1, WINDOW_T, vocab):
                raise RuntimeError(
                    f"T172 RKNN output shape mismatch: {output.shape}; "
                    f"expected {(1, WINDOW_T, vocab)}")
            if not np.issubdtype(output.dtype, np.floating):
                raise RuntimeError(
                    f"T172 RKNN output dtype mismatch: {output.dtype}; expected floating")
            output_shapes.append(list(output.shape))
            finite_full = bool(np.isfinite(output).all())
            if not finite_full:
                raise FloatingPointError("non-finite full T172 output")
            logits = output[0]
            lo = PREFIX_FRAMES + window["emit_start_local"]
            hi = min(PREFIX_FRAMES + window["emit_end_local"], logits.shape[0])
            prev = None
            ids = []
            for row in range(PREFIX_FRAMES, hi):
                token = int(np.argmax(logits[row]))
                if token != BLANK and token != prev and row >= lo:
                    ids.append(token)
                prev = token
            text_piece, detected = _detokenize(self._sp, ids)
            pieces.append(text_piece)
            detected_language = detected_language or detected
        text = "".join(pieces)
        _DECODE_METRICS.value = {
            "model_profile": MODEL_PROFILE,
            "model_sha256": MODEL_SHA256,
            "window_t": WINDOW_T, "audio_span": AUDIO_SPAN,
            "overlap": OVERLAP, "stitch_policy": "t172_center32",
            "frames_in": int(lfr.shape[0]),
            "frames_covered": int(lfr.shape[0]),
            "finite_full": finite_full, "output_shapes": output_shapes,
            "elapsed_sec": time.monotonic() - started,
        }
        return TranscriptionResult(
            text=text, language=detected_language or (lang if lang != "auto" else ""))

