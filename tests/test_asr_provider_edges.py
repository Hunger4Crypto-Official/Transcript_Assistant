"""
The ASR backends and the chain that picks between them, without a network.

Groq is driven against the loopback stub from `test_http_util`: what goes into
the multipart body, how a verbose_json reply becomes segments, what the cost
comes to, and what each failure looks like from the outside. The local backend
is driven with a fake `faster_whisper` module injected into `sys.modules`,
which also lets the tests choose whether the import succeeds at all.

The registry tests are about the two rules in its docstring: a cloud provider
is removed under a local veto, not deprioritised, and a chain that runs out of
providers fails loudly rather than reaching for one it excluded.
"""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from _fixtures import build_sandbox
from plaud_bridge import http_util
from plaud_bridge.asr import local_provider, registry
from plaud_bridge.asr.base import ASRError, ASRProvider, ASRResult
from plaud_bridge.asr.groq_provider import GroqASR
from plaud_bridge.asr.local_provider import LocalWhisperASR
from plaud_bridge.asr.registry import build_asr_chain, transcribe
from plaud_bridge.audio.prepare import AudioChunk
from plaud_bridge.config import Config
from plaud_bridge.models import Segment
from plaud_bridge.runtime import OfflineError, model_path
from test_http_util import SECRET, ScriptedServer

KEY = SECRET


@pytest.fixture
def server():
    stub = ScriptedServer()
    try:
        yield stub
    finally:
        stub.close()


def _reconfigure(tmp_path, **blocks) -> Config:
    """Rewrite the sandbox config, merging two levels deep, and reload it."""
    path = tmp_path / "config" / "pipeline.yaml"
    raw = yaml.safe_load(path.read_text())
    for block, values in blocks.items():
        raw.setdefault(block, {})
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(raw[block].get(key), dict):
                raw[block][key].update(value)
            else:
                raw[block][key] = value
    path.write_text(yaml.safe_dump(raw))
    return Config.load(tmp_path / "config", root=tmp_path)


@pytest.fixture
def groq_cfg(tmp_path, monkeypatch, server):
    """The shipped config with Groq pointed at the loopback stub and a key in the env."""
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    # Backoff is a no-op so the retry tests are instant; the attempt count is
    # what matters here and it is asserted from what the server saw.
    monkeypatch.setattr(http_util, "_sleep_backoff", lambda attempt, **kw: None)
    return _reconfigure(tmp_path, asr={"groq": {"base_url": server.url, "max_retries": 2}})


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "chunk-000.wav"
    path.write_bytes(b"RIFF" + bytes(range(256)) * 8)
    return path


class _Glossary:
    def __init__(self, hint: str):
        self.hint = hint

    def asr_prompt(self) -> str:
        return self.hint


VERBOSE_JSON = {
    "text": "Hello there. General Kenobi.",
    "language": "en",
    "duration": 7.5,
    "segments": [
        {"start": 0.0, "end": 2.5, "text": " Hello there. ", "avg_logprob": -0.21, "no_speech_prob": 0.01},
        {"start": 2.5, "end": 4.0, "text": "   ", "avg_logprob": -0.9, "no_speech_prob": 0.8},
        {"start": 4.0, "end": 7.5, "text": "General Kenobi.", "avg_logprob": -0.35, "no_speech_prob": 0.02},
    ],
}


# =========================================================================
# Groq: availability
# =========================================================================
def test_groq_is_unavailable_when_disabled_in_config(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    cfg = _reconfigure(tmp_path, asr={"groq": {"enabled": False}})

    assert GroqASR(cfg).available() == (False, "disabled in config")


def test_groq_is_unavailable_without_its_key_and_names_the_variable(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert GroqASR(cfg).available() == (False, "GROQ_API_KEY not set")

    # Whitespace is not a key.
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    assert GroqASR(cfg).available() == (False, "GROQ_API_KEY not set")


def test_groq_reads_the_key_variable_name_from_config(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("MY_GROQ", KEY)
    provider = GroqASR(cfg)
    provider.key_env = "MY_GROQ"
    assert provider.available() == (True, "ready")


def test_groq_refuses_to_transcribe_when_unavailable(sandbox, monkeypatch, audio):
    cfg, _ = sandbox
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ASRError, match="groq ASR unavailable: GROQ_API_KEY not set"):
        GroqASR(cfg).transcribe_file(audio)


# =========================================================================
# Groq: the request
# =========================================================================
def test_groq_sends_the_multipart_request_the_endpoint_expects(groq_cfg, server, audio):
    server.respond_json(VERBOSE_JSON)

    GroqASR(groq_cfg, glossary=_Glossary("Plaud, Sasson, Marcus")).transcribe_file(
        audio, offset=0.0, language="en"
    )

    assert len(server.seen) == 1
    seen = server.seen[0]
    assert seen.path == "/audio/transcriptions"
    assert seen.headers["authorization"] == f"Bearer {KEY}"

    parts = seen.multipart()
    assert parts["model"]["value"] == groq_cfg.get("asr.groq.model")
    assert parts["response_format"]["value"] == "verbose_json"
    assert parts["temperature"]["value"] == "0.0"
    assert parts["timestamp_granularities[]"]["value"] == "segment"
    assert parts["language"]["value"] == "en"
    assert parts["prompt"]["value"] == "Plaud, Sasson, Marcus"
    assert parts["file"]["filename"] == "chunk-000.wav"
    assert parts["file"]["bytes"] == audio.read_bytes()


def test_groq_omits_language_and_prompt_when_there_is_nothing_to_send(groq_cfg, server, audio):
    server.respond_json(VERBOSE_JSON)

    GroqASR(groq_cfg, glossary=_Glossary("")).transcribe_file(audio)

    parts = server.seen[0].multipart()
    assert "language" not in parts
    assert "prompt" not in parts


def test_groq_strips_whitespace_from_the_key_before_sending_it(groq_cfg, server, audio, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", f"  {KEY}\n")
    server.respond_json(VERBOSE_JSON)

    GroqASR(groq_cfg).transcribe_file(audio)

    assert server.seen[0].headers["authorization"] == f"Bearer {KEY}"


def test_groq_refuses_a_file_over_the_endpoint_cap_before_uploading_anything(groq_cfg, server, tmp_path):
    big = tmp_path / "huge.wav"
    with big.open("wb") as fh:
        fh.truncate(26 * 1024 * 1024)   # sparse; st_size is what the check reads

    with pytest.raises(ASRError) as info:
        GroqASR(groq_cfg).transcribe_file(big)

    assert "huge.wav is 26.0MB" in str(info.value)
    assert "audio.max_chunk_mb" in str(info.value)
    assert server.seen == []


# =========================================================================
# Groq: the reply
# =========================================================================
def test_groq_turns_verbose_json_into_offset_segments_and_drops_blank_ones(groq_cfg, server, audio):
    server.respond_json(VERBOSE_JSON)

    result = GroqASR(groq_cfg).transcribe_file(audio, offset=100.0, language="en")

    assert result.segments == [
        Segment(start=100.0, end=102.5, text="Hello there.", confidence=-0.21, no_speech=0.01),
        Segment(start=104.0, end=107.5, text="General Kenobi.", confidence=-0.35, no_speech=0.02),
    ]
    assert result.language == "en"
    assert result.provider == "groq"
    assert result.model == groq_cfg.get("asr.groq.model")


def test_groq_bills_the_reported_duration_at_the_configured_hourly_rate(groq_cfg, server, audio):
    server.respond_json({**VERBOSE_JSON, "duration": 3600.0})

    result = GroqASR(groq_cfg).transcribe_file(audio)

    assert result.cost_usd == pytest.approx(float(groq_cfg.get("asr.groq.usd_per_audio_hour")))


def test_groq_bills_from_the_last_segment_when_no_duration_is_reported(groq_cfg, server, audio):
    reply = {k: v for k, v in VERBOSE_JSON.items() if k != "duration"}
    server.respond_json(reply)

    result = GroqASR(groq_cfg).transcribe_file(audio, offset=50.0)

    rate = float(groq_cfg.get("asr.groq.usd_per_audio_hour"))
    # Last segment ends at 7.5s into the chunk; the offset must not inflate it.
    assert result.cost_usd == pytest.approx(7.5 / 3600.0 * rate)


def test_groq_uses_the_whole_file_text_when_no_segments_come_back(groq_cfg, server, audio):
    """A zero-width span would be undiarizable and bill nothing; give it the real length."""
    server.respond_json({"text": "  the whole thing  ", "duration": 42.0, "language": "fr"})

    result = GroqASR(groq_cfg).transcribe_file(audio, offset=10.0)

    assert result.segments == [Segment(10.0, 52.0, "the whole thing")]
    assert result.language == "fr"
    rate = float(groq_cfg.get("asr.groq.usd_per_audio_hour"))
    assert result.cost_usd == pytest.approx(42.0 / 3600.0 * rate)


def test_groq_whole_file_fallback_without_a_duration_is_zero_width_rather_than_invented(groq_cfg, server, audio):
    server.respond_json({"text": "only text", "segments": []})

    result = GroqASR(groq_cfg).transcribe_file(audio, offset=3.0)

    assert result.segments == [Segment(3.0, 3.0, "only text")]
    assert result.cost_usd == 0.0


def test_groq_returns_an_empty_result_for_silence(groq_cfg, server, audio):
    server.respond_json({"text": "", "segments": [], "duration": 0})

    result = GroqASR(groq_cfg).transcribe_file(audio, language="de")

    assert result.segments == []
    assert result.cost_usd == 0.0
    assert result.language == "de"


# =========================================================================
# Groq: failures
# =========================================================================
def test_groq_wraps_a_rejected_request_with_the_status_and_body_and_never_the_key(groq_cfg, server, audio, caplog):
    server.respond(401, '{"error": {"message": "Invalid API Key"}}')

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ASRError) as info:
            GroqASR(groq_cfg).transcribe_file(audio)

    message = str(info.value)
    assert message.startswith("groq transcription failed: HTTP 401")
    assert "Invalid API Key" in message
    assert KEY not in message
    assert KEY not in caplog.text
    # 401 is not transient; one request, no retry.
    assert len(server.seen) == 1


def test_groq_retries_a_transient_failure_and_returns_the_eventual_transcript(groq_cfg, server, audio):
    server.respond(503, "overloaded")
    server.respond_json(VERBOSE_JSON)

    result = GroqASR(groq_cfg).transcribe_file(audio)

    assert len(result.segments) == 2
    assert len(server.seen) == 2


def test_groq_honours_the_configured_retry_budget(groq_cfg, server, audio):
    for _ in range(6):
        server.respond(503, "overloaded")

    with pytest.raises(ASRError, match="HTTP 503"):
        GroqASR(groq_cfg).transcribe_file(audio)

    # asr.groq.max_retries is 2 in this fixture: the first attempt plus two.
    assert len(server.seen) == 3


def test_groq_treats_a_server_that_hangs_up_as_a_failed_call_not_a_crash(groq_cfg, server, audio):
    """The registry can only fail over on ASRError. Anything else kills the recording."""
    for _ in range(3):
        server.drop()

    with pytest.raises(ASRError, match="groq transcription failed: network error"):
        GroqASR(groq_cfg).transcribe_file(audio)


# =========================================================================
# Local: a fake faster-whisper
# =========================================================================
@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class FakeInfo:
    language: str = "en"


def fake_whisper(monkeypatch, segments=(), info=None, fail=None):
    """
    Install a stand-in `faster_whisper` and return the calls it received.

    `WhisperModel(...)` records its constructor arguments; `.transcribe(...)`
    records its keyword arguments and yields `segments`, or raises `fail`.
    """
    calls: dict[str, list] = {"ctor": [], "transcribe": []}

    class WhisperModel:
        def __init__(self, target, **kwargs):
            calls["ctor"].append((target, kwargs))

        def transcribe(self, path, **kwargs):
            calls["transcribe"].append((path, kwargs))
            if fail is not None:
                raise fail
            return iter(list(segments)), (info if info is not None else FakeInfo())

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(local_provider, "_MODEL_CACHE", {})
    return calls


def _no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)


def _torch(monkeypatch, cuda: bool):
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)


# --- availability -----------------------------------------------------------
def test_local_is_unavailable_when_disabled_in_config(tmp_path, monkeypatch):
    cfg, _ = build_sandbox(tmp_path, monkeypatch)
    fake_whisper(monkeypatch)
    # Config validation needs one enabled non-cloud provider, so disable it
    # after loading rather than in the YAML.
    cfg._d["asr"]["local"]["enabled"] = False

    assert LocalWhisperASR(cfg).available() == (False, "disabled in config")


def test_local_says_how_to_install_faster_whisper_when_it_is_missing(sandbox, monkeypatch):
    cfg, _ = sandbox
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    ok, why = LocalWhisperASR(cfg).available()

    assert ok is False
    assert "pip install faster-whisper" in why
    assert "father and husband" in why


def test_local_is_ready_once_faster_whisper_imports(sandbox, monkeypatch):
    cfg, _ = sandbox
    fake_whisper(monkeypatch)
    assert LocalWhisperASR(cfg).available() == (True, "ready")


def test_local_refuses_to_transcribe_when_faster_whisper_is_missing(sandbox, monkeypatch, audio):
    cfg, _ = sandbox
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    with pytest.raises(ASRError, match="local ASR unavailable: faster-whisper is not installed"):
        LocalWhisperASR(cfg).transcribe_file(audio)


# --- device selection ------------------------------------------------------
def test_local_auto_device_falls_back_to_cpu_int8_without_torch(sandbox, monkeypatch):
    cfg, _ = sandbox
    _no_torch(monkeypatch)
    assert LocalWhisperASR(cfg)._resolve_device() == ("cpu", "int8")


def test_local_auto_device_picks_cuda_float16_when_torch_sees_a_gpu(sandbox, monkeypatch):
    cfg, _ = sandbox
    _torch(monkeypatch, cuda=True)
    assert LocalWhisperASR(cfg)._resolve_device() == ("cuda", "float16")


def test_local_auto_device_picks_cpu_int8_when_torch_sees_no_gpu(sandbox, monkeypatch):
    cfg, _ = sandbox
    _torch(monkeypatch, cuda=False)
    assert LocalWhisperASR(cfg)._resolve_device() == ("cpu", "int8")


def test_local_auto_device_survives_a_broken_torch_install(sandbox, monkeypatch):
    """A torch that imports but explodes on `cuda` is a CPU box, not a crash."""
    cfg, _ = sandbox
    module = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", module)   # no .cuda attribute at all
    assert LocalWhisperASR(cfg)._resolve_device() == ("cpu", "int8")


def test_local_explicit_device_and_compute_type_pass_through_untouched(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    _no_torch(monkeypatch)
    cfg = _reconfigure(tmp_path, asr={"local": {"device": "cuda", "compute_type": "int8_float16"}})

    assert LocalWhisperASR(cfg)._resolve_device() == ("cuda", "int8_float16")


def test_local_explicit_cuda_with_auto_compute_gets_float16(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    _no_torch(monkeypatch)
    cfg = _reconfigure(tmp_path, asr={"local": {"device": "cuda", "compute_type": "auto"}})

    assert LocalWhisperASR(cfg)._resolve_device() == ("cuda", "float16")


# --- model loading ---------------------------------------------------------
def test_local_loads_the_configured_model_into_the_models_dir_and_allows_downloads_online(sandbox, monkeypatch):
    cfg, _ = sandbox
    _no_torch(monkeypatch)
    calls = fake_whisper(monkeypatch)

    LocalWhisperASR(cfg)._model()

    assert calls["ctor"] == [(
        cfg.get("asr.local.model"),
        {
            "device": "cpu",
            "compute_type": "int8",
            "download_root": str(model_path(cfg, "whisper")),
            "local_files_only": False,
        },
    )]


def test_local_loads_the_model_once_per_device_and_reuses_it(sandbox, monkeypatch):
    cfg, _ = sandbox
    _no_torch(monkeypatch)
    calls = fake_whisper(monkeypatch)

    provider = LocalWhisperASR(cfg)
    first = provider._model()
    second = provider._model()
    assert first is second
    assert len(calls["ctor"]) == 1

    provider.device, provider.compute_type = "cuda", "float16"
    third = provider._model()
    assert third is not first
    assert len(calls["ctor"]) == 2


def _offline(tmp_path):
    return _reconfigure(
        tmp_path,
        runtime={"offline": True},
        asr={"providers": ["local"], "groq": {"enabled": False}},
        llm={"providers": ["local"], "anthropic": {"enabled": False},
             "groq": {"enabled": False}, "local": {"enabled": True}},
    )


def test_local_offline_loads_weights_from_disk_and_forbids_downloads(tmp_path, monkeypatch):
    build_sandbox(tmp_path, monkeypatch)
    _no_torch(monkeypatch)
    calls = fake_whisper(monkeypatch)
    cfg = _offline(tmp_path)
    weights = model_path(cfg, "whisper", cfg.get("asr.local.model"))
    weights.mkdir(parents=True)

    LocalWhisperASR(cfg)._model()

    target, kwargs = calls["ctor"][0]
    assert target == str(weights)
    assert kwargs["local_files_only"] is True


def test_local_offline_without_weights_refuses_before_touching_faster_whisper(tmp_path, monkeypatch):
    """Offline means offline: name the directory it wanted instead of downloading."""
    build_sandbox(tmp_path, monkeypatch)
    _no_torch(monkeypatch)
    calls = fake_whisper(monkeypatch)
    cfg = _offline(tmp_path)

    with pytest.raises(OfflineError) as info:
        LocalWhisperASR(cfg)._model()

    assert str(model_path(cfg, "whisper", cfg.get("asr.local.model"))) in str(info.value)
    assert calls["ctor"] == []


# --- transcription ---------------------------------------------------------
def test_local_turns_faster_whisper_segments_into_offset_segments_and_drops_blank_ones(sandbox, monkeypatch, audio):
    cfg, _ = sandbox
    _no_torch(monkeypatch)
    fake_whisper(monkeypatch, segments=[
        FakeSegment(0.0, 1.5, "  How was practice?  ", -0.12, 0.01),
        FakeSegment(1.5, 2.0, "   ", -1.3, 0.9),
        FakeSegment(2.0, 4.25, "Coach said I'm starting.", -0.4, 0.03),
    ], info=FakeInfo(language="en"))

    result = LocalWhisperASR(cfg).transcribe_file(audio, offset=30.0)

    assert result.segments == [
        Segment(30.0, 31.5, "How was practice?", confidence=-0.12, no_speech=0.01),
        Segment(32.0, 34.25, "Coach said I'm starting.", confidence=-0.4, no_speech=0.03),
    ]
    assert result.language == "en"
    assert result.provider == "local"
    assert result.model == cfg.get("asr.local.model")
    assert result.cost_usd == 0.0


def test_local_passes_the_beam_size_language_and_glossary_hint_to_faster_whisper(tmp_path, monkeypatch, audio):
    build_sandbox(tmp_path, monkeypatch)
    _no_torch(monkeypatch)
    calls = fake_whisper(monkeypatch)
    cfg = _reconfigure(tmp_path, asr={"local": {"beam_size": 3}})

    LocalWhisperASR(cfg, glossary=_Glossary("Plaud, Sasson")).transcribe_file(audio, language="es")

    assert calls["transcribe"] == [(str(audio), {
        "language": "es",
        "beam_size": 3,
        "initial_prompt": "Plaud, Sasson",
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "word_timestamps": False,
    })]


def test_local_sends_no_initial_prompt_when_the_glossary_is_empty(sandbox, monkeypatch, audio):
    cfg, _ = sandbox
    _no_torch(monkeypatch)
    calls = fake_whisper(monkeypatch)

    LocalWhisperASR(cfg, glossary=_Glossary("")).transcribe_file(audio)

    _, kwargs = calls["transcribe"][0]
    assert kwargs["initial_prompt"] is None
    assert kwargs["language"] is None


def test_local_reports_the_detected_language_and_falls_back_when_none_is_reported(sandbox, monkeypatch, audio):
    cfg, _ = sandbox
    _no_torch(monkeypatch)

    fake_whisper(monkeypatch, segments=[FakeSegment(0, 1, "hola")], info=FakeInfo(language="es"))
    assert LocalWhisperASR(cfg).transcribe_file(audio).language == "es"

    fake_whisper(monkeypatch, segments=[FakeSegment(0, 1, "x")], info=object())
    assert LocalWhisperASR(cfg).transcribe_file(audio, language="fr").language == "fr"
    assert LocalWhisperASR(cfg).transcribe_file(audio).language == "en"


def test_local_wraps_a_faster_whisper_crash_with_the_file_name_and_the_real_cause(sandbox, monkeypatch, audio):
    cfg, _ = sandbox
    _no_torch(monkeypatch)
    fake_whisper(monkeypatch, fail=RuntimeError("CUDA out of memory"))

    with pytest.raises(ASRError) as info:
        LocalWhisperASR(cfg).transcribe_file(audio)

    assert str(info.value) == "faster-whisper failed on chunk-000.wav: CUDA out of memory"
    assert isinstance(info.value.__cause__, RuntimeError)


# =========================================================================
# Registry: chain construction
# =========================================================================
class _DictCfg:
    """Just enough config for the chain builder, for shapes the validator rejects."""

    def __init__(self, data: dict):
        self._d = data

    def get(self, dotted: str, default=None):
        node = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def test_chain_follows_the_configured_order(sandbox, monkeypatch):
    cfg, _ = sandbox
    chain = build_asr_chain(cfg)
    assert [p.name for p in chain] == ["groq", "local"]
    assert [p.is_cloud for p in chain] == [True, False]


def test_a_local_veto_removes_cloud_providers_rather_than_reordering_them(sandbox):
    cfg, _ = sandbox
    chain = build_asr_chain(cfg, local_only=True)
    assert [p.name for p in chain] == ["local"]
    assert all(not p.is_cloud for p in chain)


def test_an_unknown_provider_name_is_skipped_with_a_warning(tmp_path, monkeypatch, caplog):
    build_sandbox(tmp_path, monkeypatch)
    cfg = _reconfigure(tmp_path, asr={"providers": ["deepgram", "local"], "deepgram": {"enabled": True}})

    with caplog.at_level(logging.WARNING):
        chain = build_asr_chain(cfg)

    assert [p.name for p in chain] == ["local"]
    assert "unknown ASR provider 'deepgram'" in caplog.text


def test_an_empty_provider_list_builds_an_empty_chain():
    assert build_asr_chain(_DictCfg({"asr": {"providers": []}})) == []
    assert build_asr_chain(_DictCfg({})) == []


# =========================================================================
# Registry: transcribe
# =========================================================================
class _Scripted(ASRProvider):
    """A provider whose availability and per-chunk behaviour the test dictates."""

    is_cloud = False

    def __init__(self, cfg, glossary=None, *, name="scripted", ready=(True, "ready"),
                 results=None, cost=0.0):
        super().__init__(cfg, glossary)
        self.name = name
        self.model = f"{name}-model"
        self.ready = ready
        self.results = list(results or [])
        self.cost = cost
        self.calls: list[Path] = []

    def available(self):
        return self.ready

    def transcribe_file(self, path, offset=0.0, language=None):
        self.calls.append(path)
        step = self.results.pop(0)
        if isinstance(step, Exception):
            raise step
        return ASRResult(
            segments=[Segment(offset + s, offset + e, t) for s, e, t in step],
            language="en", provider=self.name, model=f"{self.name}-model", cost_usd=self.cost,
        )


def _chunks(tmp_path, n=2, length=60.0):
    out = []
    for i in range(n):
        path = tmp_path / f"chunk-{i:03d}.wav"
        path.write_bytes(b"RIFF")
        out.append(AudioChunk(i, path, i * length, length, 0.0))
    return out


def _use(monkeypatch, *providers: _Scripted):
    """Install scripted providers as the whole registry, in the given order."""
    names = [p.name for p in providers]
    monkeypatch.setattr(registry, "build_asr_chain", lambda cfg, glossary=None, local_only=False: list(providers))
    return _DictCfg({"asr": {"providers": names}})


def test_transcribe_fails_loudly_when_the_chain_is_empty_and_says_why(monkeypatch):
    monkeypatch.setattr(registry, "build_asr_chain", lambda *a, **k: [])

    with pytest.raises(ASRError) as generic:
        transcribe([], _DictCfg({}), local_only=False)
    with pytest.raises(ASRError) as vetoed:
        transcribe([], _DictCfg({}), local_only=True)

    assert "Check asr.providers in pipeline.yaml" in str(generic.value)
    assert "pip install faster-whisper" in str(vetoed.value)
    assert "Compliance requires local processing" in str(vetoed.value)


def test_transcribe_skips_an_unavailable_provider_and_uses_the_next(tmp_path, monkeypatch):
    chunks = _chunks(tmp_path, n=1)
    absent = _Scripted(None, name="absent", ready=(False, "GROQ_API_KEY not set"))
    present = _Scripted(None, name="present", results=[[(0, 5, "hello")]])
    cfg = _use(monkeypatch, absent, present)

    result = transcribe(chunks, cfg)

    assert result.asr_provider == "present"
    assert result.asr_model == "present-model"
    assert [s.text for s in result.segments] == ["hello"]
    assert absent.calls == []


def test_transcribe_falls_over_to_the_next_provider_when_one_fails_mid_run(tmp_path, monkeypatch):
    chunks = _chunks(tmp_path, n=2)
    flaky = _Scripted(None, name="flaky", cost=0.01,
                      results=[[(0, 5, "chunk zero")], ASRError("HTTP 503 from upstream")])
    steady = _Scripted(None, name="steady", cost=0.0,
                       results=[[(0, 5, "chunk zero again")], [(0, 5, "chunk one")]])
    cfg = _use(monkeypatch, flaky, steady)

    result = transcribe(chunks, cfg)

    assert result.asr_provider == "steady"
    assert [s.text for s in result.segments] == ["chunk zero again", "chunk one"]
    assert len(flaky.calls) == 2
    assert len(steady.calls) == 2


def test_transcribe_still_bills_what_the_failed_provider_charged_before_it_fell_over(tmp_path, monkeypatch):
    """Chunk 0 on the first provider was real money. Failing over does not refund it."""
    chunks = _chunks(tmp_path, n=2)
    flaky = _Scripted(None, name="flaky", cost=0.25,
                      results=[[(0, 5, "a")], ASRError("boom")])
    steady = _Scripted(None, name="steady", cost=0.10,
                       results=[[(0, 5, "a")], [(0, 5, "b")]])
    cfg = _use(monkeypatch, flaky, steady)

    result = transcribe(chunks, cfg)

    assert result.cost_usd == pytest.approx(0.25 + 0.10 + 0.10)


def test_transcribe_reports_every_provider_problem_when_all_of_them_fail(tmp_path, monkeypatch, caplog):
    chunks = _chunks(tmp_path, n=1)
    absent = _Scripted(None, name="absent", ready=(False, "disabled in config"))
    broken = _Scripted(None, name="broken", results=[ASRError("faster-whisper failed on chunk-000.wav")])
    cfg = _use(monkeypatch, absent, broken)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ASRError) as info:
            transcribe(chunks, cfg)

    message = str(info.value)
    assert message.startswith("all ASR providers failed:")
    assert "- absent: disabled in config" in message
    assert "- broken: faster-whisper failed on chunk-000.wav" in message
    assert "ASR provider broken failed, trying next" in caplog.text


def test_transcribe_stitches_chunks_onto_the_original_timeline(tmp_path, monkeypatch):
    chunks = _chunks(tmp_path, n=2, length=60.0)
    provider = _Scripted(None, name="p", results=[[(0, 10, "first")], [(0, 10, "second")]])
    cfg = _use(monkeypatch, provider)

    result = transcribe(chunks, cfg, language="en")

    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (0.0, 10.0, "first"), (60.0, 70.0, "second"),
    ]
    assert result.duration_seconds == 70.0
    assert result.language == "en"
