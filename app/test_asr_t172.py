"""Meaningful host tests for the app-local T172 backend seam."""
from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get(
    "RECAMERA_PRO_REPO", os.path.expanduser("~/project/recamera/recamera_pro")))
sys.path.insert(0, str(REPO))
# The host checkout's frozen environment does not carry the device-only
# voxedge wheel.  Supply the minimal public interface used at import time;
# all backend behavior under test remains the real kit implementation.
if "voxedge.backends.base" not in sys.modules:
    voxedge = types.ModuleType("voxedge")
    backends = types.ModuleType("voxedge.backends")
    base = types.ModuleType("voxedge.backends.base")
    class ASRBackend:  # noqa: N801
        pass
    class ASRCapability(Enum):
        OFFLINE = "offline"
        MULTI_LANGUAGE = "multi_language"
    @dataclass
    class TranscriptionResult:
        text: str
        language: str = ""
    base.ASRBackend = ASRBackend
    base.ASRCapability = ASRCapability
    base.TranscriptionResult = TranscriptionResult
    voxedge.backends = backends
    backends.base = base
    sys.modules.update({"voxedge": voxedge, "voxedge.backends": backends,
                        "voxedge.backends.base": base})
spec = importlib.util.spec_from_file_location("asr_t172", HERE / "asr_t172.py")
asr_t172 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(asr_t172)


def test_single_context_plan_covers_long_audio_once():
    for n in (167, 168, 169, 304, 305, 400):
        plan = asr_t172._window_plan(n)
        owners = [0] * n
        for window in plan:
            assert window["actual"] + window["pad"] == 168
            for frame in range(window["emit_start"], window["emit_end"]):
                owners[frame] += 1
        assert owners == [1] * n
    assert len(asr_t172._window_plan(168)) == 1


def test_load_forces_t172_even_if_parent_requests_other_shape(monkeypatch):
    seen = {}

    def parent_load(self, path, *, frames):
        seen["frames"] = frames
        return object()

    monkeypatch.setattr(asr_t172.RknnSenseVoiceBackend, "_load_one", parent_load)
    backend = object.__new__(asr_t172.T172SenseVoiceBackend)
    assert backend._load_one("candidate.rknn", frames=344) is not None
    assert seen["frames"] == 172


def test_constructor_fixes_single_model_and_woitn(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(asr_t172, "configured_inference_socket",
                        lambda: "/run/recamera/inferenced.sock")
    monkeypatch.setattr(asr_t172, "_verify_model", lambda path: None)
    monkeypatch.setattr(asr_t172, "_verify_bpe", lambda path: None)
    monkeypatch.setattr(asr_t172, "_verify_bpe_piece_size", lambda path: None)
    sdk_hashes = {
        module: hashlib.sha256(
            Path(asr_t172._module_source_path(module)).read_bytes()).hexdigest()
        for module in asr_t172.PARENT_SDK_SHA256
    }
    monkeypatch.setattr(asr_t172, "PARENT_SDK_SHA256", sdk_hashes)
    # Wrap (do not replace) the real parent constructor: on the device the
    # parent leaves self._sp = None until preload(), and a stub that sets _sp
    # hid a constructor that dereferenced it (2026-10-10 seq6 startup crash).
    real_init = asr_t172.RknnSenseVoiceBackend.__init__

    def spy_init(self, *args, **kwargs):
        seen.update(kwargs)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(asr_t172.RknnSenseVoiceBackend, "__init__", spy_init)
    backend = asr_t172.T172SenseVoiceBackend(
        str(tmp_path / "sensevoice_rv1126b_t172_scaled_hybrid.rknn"),
        "am.mvn", "embedding.npy", "tokens.bpe",
    )
    assert seen["rknn_model_short"] is None
    assert seen["textnorm"] == "woitn"
    assert backend._sp is None          # loaded only by preload()


def test_bpe_piece_size_mismatch_fails_before_parent(monkeypatch, tmp_path):
    constructed = []
    monkeypatch.setattr(asr_t172, "configured_inference_socket",
                        lambda: "/run/recamera/inferenced.sock")
    monkeypatch.setattr(asr_t172, "_verify_parent_sdk", lambda checker: None)
    monkeypatch.setattr(asr_t172, "_verify_model", lambda path: None)
    monkeypatch.setattr(asr_t172, "_verify_bpe", lambda path: None)

    class _SP:
        def load(self, path):
            return True

        def get_piece_size(self):
            return 100

    import types
    monkeypatch.setitem(sys.modules, "sentencepiece",
                        types.SimpleNamespace(SentencePieceProcessor=_SP))
    monkeypatch.setattr(asr_t172.RknnSenseVoiceBackend, "__init__",
                        lambda self, *a, **k: constructed.append(1))
    with pytest.raises(asr_t172.ConfigurationError) as err:
        asr_t172.T172SenseVoiceBackend(
            str(tmp_path / "m.rknn"), "am.mvn", "embedding.npy", "tokens.bpe")
    assert "piece size" in str(err.value)
    assert constructed == []


def test_parent_sdk_default_source_hash_checker_matches_receipt():
    # Uses inspect.getsourcefile(module), then hashes that actual source path.
    asr_t172._verify_parent_sdk()


def test_parent_sdk_source_hash_mismatch_fails_closed(monkeypatch):
    bad = dict(asr_t172.PARENT_SDK_SHA256)
    bad["kit.asr_rknn_backend"] = "0" * 64
    monkeypatch.setattr(asr_t172, "PARENT_SDK_SHA256", bad)
    with pytest.raises(Exception, match="source hash mismatch"):
        asr_t172._verify_parent_sdk()


def test_bpe_identity_rejects_wrong_tokenizer(tmp_path):
    path = tmp_path / "wrong.bpe"
    path.write_bytes(b"wrong tokenizer")
    with pytest.raises(Exception, match="BPE bytes/hash"):
        asr_t172._verify_bpe(str(path))


def test_empty_audio_returns_without_frontend_or_inference():
    backend = _backend_with_logits(np.zeros((172, 4), dtype=np.float32))
    called = {"prep": 0}
    backend._prep = lambda audio, lang: called.__setitem__("prep", 1)
    result = backend._transcribe_array_impl(np.zeros(0, dtype=np.float32), "en")
    assert result.text == ""
    assert result.language == "en"
    assert called["prep"] == 0
    assert backend._rknn.shapes == []


class _SentencePiece:
    pieces = {1: "<|en|>", 2: "▁hello", 3: "▁world"}

    def id_to_piece(self, token):
        return self.pieces[token]

    def get_piece_size(self):
        return 4


class _Runtime:
    def __init__(self, logits):
        self.logits = logits
        self.shapes = []

    def inference(self, *, inputs):
        self.shapes.append(inputs[0].shape)
        return [np.asarray([self.logits])]


def _backend_with_logits(logits):
    backend = object.__new__(asr_t172.T172SenseVoiceBackend)
    backend._inference_service = None
    backend._rknn = _Runtime(logits)
    backend._sp = _SentencePiece()
    backend._ensure_lease_alive = lambda: None
    backend._prep = lambda audio, lang: np.zeros((172, 560), dtype=np.float32)
    return backend


def test_transcribe_uses_full_t172_and_preserves_piece_whitespace():
    logits = np.zeros((172, 4), dtype=np.float32)
    for row, token in ((4, 1), (5, 2), (6, 2), (7, 0), (8, 3)):
        logits[row, token] = 10.0
    result = _backend_with_logits(logits)._transcribe_array_impl(
        np.zeros(16000, dtype=np.float32), "auto")
    assert result.text == " hello world"
    assert result.language == "en"


def test_decode_metrics_are_thread_local_contract():
    logits = np.zeros((172, 4), dtype=np.float32)
    logits[5, 2] = 10.0
    backend = _backend_with_logits(logits)
    backend._transcribe_array_impl(np.zeros(16000, dtype=np.float32), "auto")
    metrics = asr_t172.get_last_decode_metrics()
    assert metrics["model_profile"] == "t172_scaled_hybrid"
    assert metrics["model_sha256"] == asr_t172.MODEL_SHA256
    assert metrics["frames_covered"] == metrics["frames_in"]
    assert metrics["finite_full"] is True
    assert metrics["output_shapes"] == [[1, 172, 4]]


def test_nonfinite_full_output_fails_closed():
    logits = np.zeros((172, 4), dtype=np.float32)
    logits[20, 1] = np.nan
    with pytest.raises(FloatingPointError):
        _backend_with_logits(logits)._transcribe_array_impl(
            np.zeros(16000, dtype=np.float32), "auto")


@pytest.mark.parametrize("shape", [(1, 171, 4), (2, 172, 4), (1, 172, 5)])
def test_invalid_rknn_output_shape_fails_closed(shape):
    backend = _backend_with_logits(np.zeros(shape, dtype=np.float32))
    with pytest.raises(RuntimeError, match="output shape"):
        backend._transcribe_array_impl(
            np.zeros(16000, dtype=np.float32), "auto")


def test_integer_output_dtype_fails_closed():
    backend = _backend_with_logits(np.zeros((172, 4), dtype=np.int32))
    with pytest.raises(RuntimeError, match="output dtype"):
        backend._transcribe_array_impl(
            np.zeros(16000, dtype=np.float32), "auto")


def test_production_requires_managed_socket(monkeypatch, tmp_path):
    monkeypatch.setattr(asr_t172, "configured_inference_socket", lambda: None)
    monkeypatch.setattr(asr_t172, "_verify_parent_sdk", lambda checker=None: None)
    monkeypatch.setattr(asr_t172, "_verify_model", lambda path: None)
    with pytest.raises(Exception, match="managed inference"):
        asr_t172.T172SenseVoiceBackend(
            str(tmp_path / "model.rknn"), "cmvn", "embedding", "bpe")


def test_production_rejects_parent_injection(monkeypatch, tmp_path):
    monkeypatch.setattr(asr_t172, "configured_inference_socket",
                        lambda: "/run/recamera/inferenced.sock")
    with pytest.raises(Exception, match="injection"):
        asr_t172.T172SenseVoiceBackend(
            str(tmp_path / "model.rknn"), "cmvn", "embedding", "bpe",
            runtime_factory=object())

