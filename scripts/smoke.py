#!/usr/bin/env python3
"""
Prove the program works from outside itself.

The unit suite calls `main(argv)` in-process with a stubbed LLM, which is the
right way to test behaviour but says nothing about whether a person who just
cloned this repository can run `python run.py digest` and get a document. The
things that break on a clean machine break below the level the unit suite
looks at: an import that only resolves because pytest put `src` on the path, a
route that needs a terminal, an artifact written somewhere the reader does not
look.

So this stands up a complete throwaway project in a temp directory and drives
the real command line in real subprocesses, once per route, the way a shell
would. It needs no network, no ffmpeg, no model weights, no HuggingFace token
and no API keys, because of three arrangements:

  - The fixtures are TRANSCRIPTS, not audio. `ingest.text_extensions` accepts
    .txt and .srt and the pipeline skips ASR entirely for them, so the whole
    run happens with no ffmpeg and no Whisper weights on disk.
  - The only enabled LLM is an OpenAI-compatible stub served on loopback by
    this process. It reads the schema the extractor sent and answers in that
    shape, so the analysis is real code doing real work against a fake model.
  - Every cloud provider is disabled in the generated config and
    `runtime.offline` is on, which makes the config itself refuse to load if
    that ever stops being true.

The honesty property this suite exists for: the route list is read from
`build_parser()` at runtime, not hardcoded. A subcommand that exists in the
parser and has no entry in ROUTES is reported as a failure ("route not
covered") rather than quietly going untested. Adding a route to the program is
therefore a change that this file has to acknowledge.

    python scripts/smoke.py                 run everything, print a table
    python scripts/smoke.py --quick         one check per route, for CI
    python scripts/smoke.py --only digest   one route, everything it accepts
    python scripts/smoke.py --list          what would run, without running it
    python scripts/smoke.py --keep          leave the temp project behind
    python scripts/smoke.py --json          machine-readable result

A route reports OK, FAIL, or SKIPPED with a one-line reason. Nothing is ever
silently passed: a route that genuinely cannot run headlessly says why, in the
table, every time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plaud_bridge.cli import build_parser  # noqa: E402
from plaud_bridge.db import Database  # noqa: E402

RUN_PY = ROOT / "run.py"
REAL_DATA = ROOT / "data"

OK, FAIL, SKIP = "OK", "FAIL", "SKIPPED"

# Per-command wall clock. Generous, because a cold interpreter plus scrypt key
# derivation on a loaded CI runner is slower than it looks; short enough that a
# command which has actually hung does not hold the suite forever.
COMMAND_TIMEOUT = 180


# =========================================================================
# Fixtures
#
# Four transcripts, chosen so that between them they populate every state the
# read-only routes need to have something to say: an encrypted work recording
# with detectable consent, a personal one, a recording nothing routes, and one
# the compliance gate refuses. The consent lines are the load-bearing part of
# the first one -- insurance_agent sets require_consent, so without them the
# gate quarantines it and half the suite has no recording to read.
# =========================================================================
CLIENT_CALL = """\
Sasson: Morning Dana, before we start I record these calls for my own notes. Is that okay?
Dana: Yes, that's fine with me.
Sasson: Appreciate it. Tell me what coverage you have in place today.
Dana: Just a small term policy through work. Two hundred thousand, I think.
Sasson: And is there a disability policy anywhere, through the employer or private?
Dana: No, nothing like that.
Sasson: That is the gap I would look at first. Your income is the asset, not the house.
Dana: What drives the premium on something like that?
Sasson: The elimination period mostly, and whether you want an own occupation definition.
Dana: The cost is my worry, honestly.
Sasson: Understood, and I would rather right-size it than oversell you. Let me build two options.
Dana: Send them over and I will look at them this weekend.
Sasson: I will have them to you Thursday. Can you email me your date of birth before then?
Dana: Yes, tonight.
"""

# The same conversation with the consent exchange removed. The gate has to
# quarantine this one, which is what gives `release` something to release.
NO_CONSENT_CALL = "\n".join(CLIENT_CALL.splitlines()[2:]) + "\n"

# SRT rather than plain text, so the other accepted transcript format is
# exercised by the run rather than assumed to work.
FAMILY_DINNER_SRT = """\
1
00:00:00,000 --> 00:00:04,000
Sasson: How was practice today?

2
00:00:04,000 --> 00:00:08,500
Kid: Coach said I am starting on Saturday.

3
00:00:08,500 --> 00:00:13,000
Sasson: That is great. What time do you need to be at school for the bus?

4
00:00:13,000 --> 00:00:18,000
Kid: Nine. And you have to sign the permission slip for the field trip.

5
00:00:18,000 --> 00:00:22,000
Sasson: I will sign it tonight. Remind me at bedtime.

6
00:00:22,000 --> 00:00:26,000
Kid: Can we get pizza after the game like you said?

7
00:00:26,000 --> 00:00:29,000
Sasson: I did say that. Pizza after the game.
"""

# Deliberately unroutable: no profile's keywords, and the stub scores it low
# everywhere, so it lands in the fallback profile. `review` reads the unfiled
# pile, and with nothing in it that section of the report is never exercised.
EVENING_NOTES = """\
Sasson: Reminder to myself about the fence panel that came loose in the wind.
Sasson: The hardware shop on Fourth closes at six on weekdays.
Sasson: Also the car needs its inspection sticker before the end of the month.
"""

FIXTURES = {
    "client-factfind.txt": CLIENT_CALL,
    "family-dinner.srt": FAMILY_DINNER_SRT,
    "evening-notes.txt": EVENING_NOTES,
    "no-consent-call.txt": NO_CONSENT_CALL,
}


# =========================================================================
# The stub model
#
# An OpenAI-compatible chat endpoint on loopback. It is not a mock in the unit
# test sense: the CLI subprocess builds a real request, sends it over a real
# socket through the real provider class, and parses a real response. What is
# fake is only the intelligence.
# =========================================================================
_SCHEMA_FIELD_RE = re.compile(
    r'^\s*"(?P<key>[A-Za-z_][A-Za-z0-9_]*)":\s*(?P<type>list\[[a-z]+\]|string|boolean|int|integer|float|number)',
    re.MULTILINE,
)
_CATALOGUE_ID_RE = re.compile(r"^\s*-\s*id:\s*(?P<id>[A-Za-z_][A-Za-z0-9_]*)\s*$", re.MULTILINE)

# Which profile a fixture should route to, keyed on a phrase only that fixture
# contains. Keeping the mapping here rather than in the handler means the
# routing answers stay readable next to the transcripts they describe.
_ROUTING_MARKERS = (
    ("elimination period", "insurance_agent"),
    ("permission slip", "father"),
)


def _routing_reply(user: str) -> dict[str, Any]:
    """Score every profile the router offered, from the transcript it sent."""
    lowered = user.lower()
    winner = next((pid for marker, pid in _ROUTING_MARKERS if marker in lowered), "")
    # The catalogue is read out of the prompt rather than hardcoded, so a
    # profile added to config/ (or scaffolded by the new-profile route mid-run)
    # is scored rather than ignored.
    ids = _CATALOGUE_ID_RE.findall(user)
    return {
        "scores": [
            {
                "profile_id": pid,
                "score": 0.95 if pid == winner else 0.02,
                "evidence": ["smoke fixture"] if pid == winner else [],
            }
            for pid in ids
        ]
    }


def _ask_reply(_user: str) -> dict[str, Any]:
    """
    Answer the question route in the shape its output contract asks for.

    The citation list is left empty on purpose. A citation naming a recording
    that was not in the excerpts is dropped by `ask` and turns into a caveat on
    the answer, so a stub that invented one would be manufacturing the very
    warning the suite would then be reading.
    """
    return {
        "answer": "The smoke fixture answers this question.",
        "citations": [],
        "confidence": "medium",
        "unanswered": "",
    }


def _extraction_reply(user: str) -> dict[str, Any]:
    """
    Answer in the shape the profile's own schema asked for.

    Parsing the schema back out of the prompt keeps this stub correct when a
    profile gains a field: the reply grows with it, so the smoke run keeps
    exercising the coercion and rendering paths for every configured field
    rather than only the ones that existed when this was written.
    """
    schema = user.split("TRANSCRIPT:", 1)[0]
    payload: dict[str, Any] = {}
    for match in _SCHEMA_FIELD_RE.finditer(schema):
        key, kind = match.group("key"), match.group("type")
        if kind == "boolean":
            # Never true. `requires_human_attention` makes the extractor discard
            # every other field, which would empty the digest and make the
            # rendering routes pass without rendering anything.
            payload[key] = False
        elif kind == "list[quote]":
            payload[key] = [
                {"timestamp": "00:08", "speaker": "Sasson", "text": "smoke fixture quote"}
            ]
        elif kind == "list[object]":
            payload[key] = [{"what": "smoke fixture item", "when": "Saturday"}]
        elif kind.startswith("list"):
            payload[key] = ["smoke fixture item"]
        elif kind in ("int", "integer"):
            payload[key] = 0
        elif kind in ("float", "number"):
            payload[key] = 0.0
        else:
            payload[key] = "smoke fixture value"
    return payload


class _StubServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler) -> None:
        super().__init__(address, handler)
        self.completions = 0
        self.counter_lock = threading.Lock()


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - the name is BaseHTTPRequestHandler's
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length)
        if not self.path.endswith("/chat/completions"):
            self.send_error(404, "only /chat/completions is stubbed")
            return
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400, "request body was not JSON")
            return

        user = "".join(
            str(m.get("content", ""))
            for m in request.get("messages", [])
            if m.get("role") == "user"
        )
        # Three prompts reach this endpoint and each wants a different shape:
        # the router asks for scores, `ask` asks for an answer with citations,
        # and extraction asks for the profile's own schema.
        if '"scores"' in user:
            payload = _routing_reply(user)
        elif user.startswith("QUESTION:") and "EXCERPTS:" in user:
            payload = _ask_reply(user)
        else:
            payload = _extraction_reply(user)
        body = json.dumps({
            "id": "smoke",
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": json.dumps(payload)}}],
            # Both rates are zero in the generated config, so this reports work
            # done without inventing spend the cost guardrail would act on.
            "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 64},
        }).encode("utf-8")

        server: _StubServer = self.server  # type: ignore[assignment]
        with server.counter_lock:
            server.completions += 1

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence. The suite's own output is the report."""


# =========================================================================
# The throwaway project
# =========================================================================
@dataclass
class Sandbox:
    root: Path
    config_dir: Path
    out_dir: Path
    env: dict[str, str]
    server: _StubServer
    thread: threading.Thread

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def build_sandbox(root: Path) -> Sandbox:
    """Copy the shipped config into `root`, point it at loopback, and serve it."""
    server = _StubServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    config_dir = root / "config"
    shutil.copytree(ROOT / "config", config_dir)

    data = yaml.safe_load((config_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    for key in ("inbox", "work", "outbox", "vault", "quarantine", "logs"):
        data["paths"][key] = str(root / "data" / key)
    data["paths"]["database"] = str(root / "data" / "bridge.db")

    # Diarization needs pyannote weights and a HuggingFace token. Off here; the
    # documented degraded mode is a single unlabelled speaker track, and the
    # speaker labels this suite relies on come from the transcript text itself.
    data["diarization"]["enabled"] = False
    # Files are written and processed in the same second, so the settle window
    # would skip every one of them.
    data["ingest"]["settle_seconds"] = 0

    data["asr"]["providers"] = ["local"]
    data["asr"]["groq"]["enabled"] = False
    data["llm"]["providers"] = ["local"]
    data["llm"]["anthropic"]["enabled"] = False
    data["llm"]["groq"]["enabled"] = False
    data["llm"]["local"].update({
        "enabled": True,
        "is_cloud": False,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "model": "smoke-stub",
        "timeout_seconds": 30,
        # One attempt. A stub that fails should surface immediately rather than
        # spending the backoff schedule proving it will keep failing.
        "max_retries": 0,
    })

    # Offline is an assertion the config enforces at load: with any cloud
    # provider still enabled it refuses to start. Turning it on here means the
    # generated project would fail loudly if a future edit re-enabled one,
    # instead of this suite quietly acquiring a network dependency.
    data["runtime"]["offline"] = True

    (config_dir / "pipeline.yaml").write_text(yaml.safe_dump(data, sort_keys=False),
                                              encoding="utf-8")

    inbox = root / "data" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for name, body in FIXTURES.items():
        (inbox / name).write_text(body, encoding="utf-8")

    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A file that exists and is not a recording. Several routes check that a
    # path exists before they reach anything needing ffmpeg or model weights,
    # and that early branch is worth exercising even when the rest is not.
    (out_dir / "not-a-recording.wav").write_bytes(b"not audio, deliberately")

    env = dict(os.environ)
    # A fresh passphrase per run. The vault is real, the encryption is real, and
    # nothing that survives this process can open it.
    env["PLAUD_BRIDGE_PASSPHRASE"] = "smoke-" + secrets.token_urlsafe(24)
    # Credentials that happen to be in the operator's shell must not change what
    # this suite does. An empty value reads as "not set" everywhere it matters.
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "HUGGINGFACE_TOKEN"):
        env[key] = ""
    # A proxy in the environment would otherwise be asked to fetch loopback.
    for key in ("NO_PROXY", "no_proxy"):
        env[key] = ",".join(filter(None, [env.get(key, ""), "127.0.0.1", "localhost"]))
    env["PYTHONIOENCODING"] = "utf-8"

    return Sandbox(root, config_dir, out_dir, env, server, thread)


# =========================================================================
# Routes and their checks
# =========================================================================
@dataclass(frozen=True)
class Check:
    """One invocation of the CLI, and what it has to do."""

    argv: tuple[str, ...]
    expect: tuple[int, ...] = (0,)
    contains: str = ""
    creates: str = ""
    stdin: str = ""
    quick: bool = False
    skip: str = ""

    @property
    def label(self) -> str:
        return " ".join(self.argv)


def c(*argv: str, **kwargs: Any) -> Check:
    return Check(tuple(argv), **kwargs)


# The coverage table. Every key must exist in the parser and every parser route
# must appear here; both directions are checked at the end of a full run.
#
# Placeholders are filled in after the fixtures have been processed:
#   {rec}    a completed, encrypted work recording
#   {held}   the recording the compliance gate quarantined
#   {doomed} the recording `forget` is allowed to destroy
#   {out}    a scratch directory inside the throwaway project
ROUTES: dict[str, list[Check]] = {
    # doctor reports a missing dependency as exit 1. On a machine with no
    # ffmpeg and no Whisper weights that is the correct answer, not a failure,
    # so what is asserted is that it produced a verdict rather than a traceback.
    "doctor": [
        c("doctor", expect=(0, 1), contains="RESULT:", quick=True),
        c("doctor", "--offline", expect=(0, 1), contains="offline:providers"),
    ],
    "run": [
        c("run", "--limit", "1", contains="processed=1", quick=True),
        c("run", contains="failed=0", quick=True),
        # Dedupe is the reason `run` is safe on a cron: the second pass must
        # process nothing and still exit clean.
        c("run", contains="processed=0"),
        c("run", "--force"),
    ],
    "status": [c("status", contains="recordings", quick=True)],
    "digest": [
        c("digest", quick=True),
        c("digest", "--days", "30"),
        c("digest", "--profile", "insurance_agent"),
        c("digest", "--profile", "father", "--days", "30"),
        c("digest", "--include-personal"),
        c("digest", "--title", "Smoke"),
        c("digest", "--out", "{out}/digest.md", creates="{out}/digest.md"),
        c("digest", "--format", "html", "--out", "{out}/digest.html",
          creates="{out}/digest.html"),
    ],
    # Exit 2 from a content search means "the answer is incomplete", which is an
    # answer and not a crash. Only --scan-limit should produce it here. The
    # quarantined fixture used to as well, permanently, because the archive
    # reported a recording it had deliberately never written as one it could not
    # open; these checks demand 0 so that regression cannot come back quietly.
    "search": [
        c("search", "factfind", quick=True),
        c("search", "elimination period", "--content", contains="hit(s)"),
        c("search", "elimination period", "--content", "--context", "2"),
        c("search", "elimination period", "--content", "--scan-limit", "1", expect=(0, 2)),
        c("search", "coverage", "--content", "--per-recording", "1"),
        c("search", "nothing was ever said about this", "--content"),
        c("search", "factfind", "--profile", "insurance_agent", "--days", "30", "--limit", "5"),
    ],
    "open": [
        c("open", "{rec}", quick=True),
        c("open", "{rec}", "--kind", "transcript", "--out", "{out}/t.md",
          creates="{out}/t.md"),
        c("open", "{rec}", "--kind", "analysis", "--out", "{out}/a.json",
          creates="{out}/a.json"),
        c("open", "{rec}", "--kind", "source", "--out", "{out}/s.txt",
          creates="{out}/s.txt"),
        c("open", "{rec}", "--kind", "audio",
          skip="a transcript import has no audio artifact; producing one needs "
               "ffmpeg and a real recording"),
        c("open", "rec_does_not_exist", expect=(1,)),
    ],
    "verify": [c("verify", contains="artifact(s) indexed", quick=True)],
    # `ask` exits 2 when the answer is incomplete, for the same reason
    # `search --content` does. A quarantined recording is not one of those
    # reasons: it holds nothing to read, which is different from something
    # unreadable, and these checks demand 0 to keep it that way.
    "ask": [
        c("ask", "what did I promise Dana?", quick=True),
        c("ask", "what did I promise Dana?", "--profile", "insurance_agent"),
        c("ask", "what is outstanding?", "--days", "30", "--limit", "5"),
        c("ask", "what did we agree at home?", "--include-personal"),
        c("ask", "what did I promise Dana?", "--local-only"),
        c("ask", "what did I promise Dana?", "--save", contains="saved, encrypted"),
        # A question the archive has nothing to say about is a complete answer
        # that happens to be "nothing", and exits 0 like the search that backs
        # it. A profile that does not exist is the asker's mistake, and exits 1.
        c("ask", "nothing was ever said about this at all"),
        c("ask", "anything", "--profile", "no-such-profile", expect=(1,),
          contains="no profile called"),
    ],
    # Exit 2 means "some recordings were omitted because they would not open",
    # and the fixture archive contains a quarantined recording, which is exactly
    # the case that used to trigger it wrongly: the gate never writes that
    # recording's content anywhere, and reporting "could not be decrypted" for a
    # thing that was deliberately never encrypted sent people after a passphrase
    # problem they did not have. These demand 0 so the distinction stays real.
    "export": [
        c("export", quick=True),
        c("export", "--transcripts"),
        c("export", "--include-personal"),
        c("export", "--title", "Handover"),
        c("export", "--profile", "insurance_agent", "--days", "30", "--limit", "5"),
        # A personal profile is refused unless it is asked for explicitly.
        c("export", "--profile", "father", expect=(1,)),
        c("export", "--out", "{out}/export.md", creates="{out}/export.md"),
        c("export", "--format", "html", "--out", "{out}/export.html",
          creates="{out}/export.html"),
        c("export", "--transcripts", "--format", "html", "--out", "{out}/export-t.html",
          creates="{out}/export-t.html"),
    ],
    "audit": [
        c("audit", quick=True),
        c("audit", "--action", "ingest"),
        c("audit", "--actor", "pipeline"),
        c("audit", "--recording-id", "{rec}"),
        c("audit", "--days", "7", "--limit", "5"),
        c("audit", "--out", "{out}/audit.csv", creates="{out}/audit.csv"),
    ],
    "followups": [
        c("followups", quick=True),
        c("followups", "--status", "all"),
        c("followups", "--status", "done"),
        c("followups", "--status", "dropped"),
        c("followups", "--profile", "insurance_agent"),
        c("followups", "--days", "30"),
        c("followups", "--include-personal"),
        c("followups", "--title", "Open Items"),
        c("followups", "--out", "{out}/followups.md", creates="{out}/followups.md"),
        c("followups", "--format", "html", "--out", "{out}/followups.html",
          creates="{out}/followups.html"),
        # A draft is written, never sent. All three ways of naming what to draft
        # are exercised: everything open, one recording's debts, and one item.
        c("followups", "--draft", "open", contains="wrote"),
        c("followups", "--draft", "{rec}", contains="wrote"),
        c("followups", "--draft", "{followup}", "--format", "text", contains="wrote"),
        c("followups", "--done", "{followup}", contains="is now done"),
        c("followups", "--reopen", "{followup}", contains="is now open"),
        c("followups", "--drop", "{followup}", contains="is now dropped"),
        c("followups", "--done", "fu_000000000000", expect=(1,)),
    ],
    "review": [
        c("review", quick=True),
        c("review", "--days", "7"),
        # The reaffirmation prompt is the one place a route wants a human. Feed
        # it the word it asks for rather than skipping the route.
        c("review", "--reaffirm", "father", stdin="YES\n", contains="recorded"),
        c("review", "--reaffirm", "insurance_agent", expect=(1,)),
    ],
    "release": [
        c("release", "{held}", "--yes", contains="released", quick=True),
        c("release", "rec_does_not_exist", "--yes", expect=(1,)),
    ],
    "watch": [
        # `watch` with no bound runs until interrupted, which is what it is for
        # and not something a suite can assert. The bounded forms are the whole
        # of the route otherwise.
        c("watch", "--once", quick=True),
        c("watch", "--max-runs", "1", "--interval", "1"),
    ],
    "profiles": [c("profiles", contains="profile(s)", quick=True)],
    "new-profile": [
        c("new-profile", "smoke_mentor", "--name", "Smoke Mentor", "--short-name", "Mentor",
          "--heading", "Mentoring", contains="wrote", quick=True),
        # The scaffold has to load, so this reads the routing table back.
        c("profiles", contains="Smoke Mentor"),
        c("new-profile", "smoke_mentor", expect=(1,)),
    ],
    "voices": [c("voices", contains="active voice", quick=True)],
    # Memory is derived from what the run already stored, so by the time this
    # route is reached the ledgers exist. --rebuild is the interesting one: it
    # throws them away and replays the archive, and it is the answer to believe
    # whenever the ledger and the archive disagree.
    "memory": [
        c("memory", contains="insurance_agent", quick=True),
        c("memory", "--profile", "insurance_agent"),
        c("memory", "--brief"),
        c("memory", "--rebuild", contains="replayed"),
        # A profile that does not exist is a typo, and a typo that prints an
        # empty ledger reads as "nothing was recorded" rather than "no such
        # profile". It exits 1 and says which profiles are real.
        c("memory", "--profile", "no-such-profile", expect=(1,), contains="unknown profile"),
        # Removing a recording from the ledgers is the half of `forget` that
        # would otherwise keep feeding a deleted conversation into prompts.
        c("memory", "--forget", "{doomed}", contains="removed"),
    ],
    # The speaker group is the one place where the useful work genuinely cannot
    # happen here: a voiceprint is an embedding of real speech, which needs
    # ffmpeg to prepare a clip and the pyannote embedding weights to turn it
    # into a vector. What is reachable is every route's argument handling and
    # its behaviour on an empty store, which is the state a new install is in.
    "speakers list": [c("speakers", "list", contains="Nobody is enrolled", quick=True)],
    "speakers enroll": [
        c("speakers", "enroll", "Smoke Person", "--audio", "{out}/no-such-clip.wav",
          expect=(1,), contains="no such file", quick=True),
        c("speakers", "enroll", "Smoke Person", "--audio", "{out}/not-a-recording.wav",
          "--start", "0", "--end", "5", "--replace",
          skip="enrolling a voice needs ffmpeg and the pyannote embedding weights"),
    ],
    "speakers identify": [
        c("speakers", "identify", "{out}/no-such-clip.wav", expect=(1,),
          contains="no such file", quick=True),
        c("speakers", "identify", "{out}/not-a-recording.wav", expect=(1,),
          contains="Nobody is enrolled"),
        c("speakers", "identify", "{out}/not-a-recording.wav",
          skip="scoring a recording needs ffmpeg, diarization, and an enrolled voice"),
    ],
    "speakers forget": [
        c("speakers", "forget", "Nobody At All", "--yes", expect=(1,), quick=True),
        c("speakers", "forget", "Smoke Person", "--yes",
          skip="there is nothing to delete without an enrollment, which needs the "
               "embedding weights"),
    ],
    # A backup is one encrypted file, so writing one for real is cheap enough
    # to do here. The default output path is the operator's home directory,
    # which a suite promising to leave the machine alone must never touch, so
    # every check passes --out.
    "backup": [
        c("backup", "--out", "{out}/backup.pbb", contains="wrote",
          creates="{out}/backup.pbb", quick=True),
    ],
    # The route brings its own backup file, so `--only restore` stands alone.
    # The refusal comes first: this sandbox already holds data, which is
    # exactly what a careless restore meets, and the answer has to be "no,
    # unless you say --force".
    "restore": [
        c("backup", "--out", "{out}/restore-fixture.pbb", contains="wrote", quick=True),
        c("restore", "{out}/restore-fixture.pbb", expect=(1,), contains="--force",
          quick=True),
        c("restore", "{out}/restore-fixture.pbb", "--force", contains="restored"),
        c("restore", "{out}/no-such-backup.pbb", expect=(1,)),
    ],
    "retention": [
        c("retention", quick=True),
        c("retention", "--execute", "--yes"),
    ],
    "forget": [
        # With stdin closed there is nobody to type FORGET, and the only safe
        # reading of no answer is no. Checked before the real delete, because
        # afterwards there would be nothing left to refuse to delete.
        c("forget", "{doomed}", expect=(1,)),
        c("forget", "{doomed}", "--yes", contains="deleted", quick=True),
        c("forget", "rec_does_not_exist", "--yes", expect=(1,)),
    ],
}

# Execution order. Reading routes come before writing ones, and the two that
# destroy something come last, so an earlier route never fails because a later
# one already deleted what it wanted to read. `run` is first because every
# other route needs what it produces.
ROUTE_ORDER = (
    "doctor", "run", "status", "profiles", "voices", "memory", "digest", "search", "open",
    "verify", "ask", "export", "audit", "followups", "review", "backup", "restore",
    "speakers list", "speakers enroll",
    "speakers identify", "speakers forget", "release", "watch", "new-profile",
    "retention", "forget",
)


def parser_routes() -> list[str]:
    """
    Every subcommand a person can reach, read from the parser itself.

    argparse exposes no public accessor for the subcommand table, so this walks
    `_actions` the way tests/test_cli_routes.py does. Nested groups are walked
    too and reported as "parent child", so a command group added later shows up
    as several routes rather than one unreachable name.
    """
    def walk(parser: argparse.ArgumentParser, prefix: str = "") -> list[str]:
        found: list[str] = []
        for action in parser._actions:
            choices = getattr(action, "choices", None)
            if not isinstance(choices, dict):
                continue
            for name, sub in choices.items():
                if not isinstance(sub, argparse.ArgumentParser):
                    continue
                nested = walk(sub, f"{prefix}{name} ")
                found.extend(nested or [f"{prefix}{name}"])
        return found

    return sorted(set(walk(build_parser())))


# =========================================================================
# Running
# =========================================================================
@dataclass
class Result:
    label: str
    status: str
    detail: str = ""
    exit_code: int | None = None
    seconds: float = 0.0


@dataclass
class RouteResult:
    route: str
    status: str
    checks: list[Result] = field(default_factory=list)
    note: str = ""

    @property
    def passed(self) -> int:
        return len([r for r in self.checks if r.status == OK])

    @property
    def ran(self) -> int:
        return len([r for r in self.checks if r.status != SKIP])


class Runner:
    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self.subs: dict[str, str] = {"out": str(sandbox.out_dir)}

    def invoke(self, argv: list[str], stdin: str = "") -> tuple[int, str]:
        command = [sys.executable, str(RUN_PY), "--config", str(self.sandbox.config_dir), *argv]
        try:
            completed = subprocess.run(
                command,
                cwd=self.sandbox.root,
                env=self.sandbox.env,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {COMMAND_TIMEOUT}s"
        return completed.returncode, completed.stdout + completed.stderr

    def refresh_fixture_ids(self) -> None:
        """
        Find the identifiers the later routes need.

        Recording ids come from the index rather than from stdout, because they
        are an implementation detail of the run and a suite that parses its own
        subject's prose starts failing when the prose is reworded. The follow-up
        id below is the exception, and it earns it.
        """
        db = Database(self.sandbox.root / "data" / "bridge.db")
        try:
            complete = [r for r in db.query(limit=50) if r["stage"] == "complete"]
            held = db.query(stage="quarantined", limit=50)
        finally:
            db.close()

        work = [r for r in complete if r["governing_profile"] == "insurance_agent"]
        spare = [r for r in complete if r["governing_profile"] != "insurance_agent"]
        if work:
            self.subs["rec"] = work[0]["id"]
        if held:
            self.subs["held"] = held[0]["id"]
        # Whatever `forget` destroys must not be something an earlier route
        # still needs, so it is never the work recording the read routes use.
        doomed = spare or work
        if doomed:
            self.subs["doomed"] = doomed[-1]["id"]

        # A follow-up id is not in the index. It is derived from the analyses
        # and printed by the route itself, beside the command that consumes it,
        # so reading one back out of that output is exactly what a person does.
        _code, output = self.invoke(["followups"])
        found = re.search(r"--done\s+(fu_[0-9a-f]+)", output)
        if found:
            self.subs["followup"] = found.group(1)

    def render(self, value: str) -> str | None:
        try:
            return value.format(**self.subs)
        except KeyError:
            return None

    def run_check(self, check: Check) -> Result:
        if check.skip:
            return Result(check.label, SKIP, check.skip)

        argv: list[str] = []
        for token in check.argv:
            rendered = self.render(token)
            if rendered is None:
                return Result(
                    check.label, FAIL,
                    f"the run produced no fixture for '{token}', so this could not be tried",
                )
            argv.append(rendered)

        started = time.monotonic()
        code, output = self.invoke(argv, stdin=check.stdin)
        elapsed = time.monotonic() - started
        label = " ".join(argv)

        if code not in check.expect:
            detail = f"exit {code}, expected {' or '.join(str(e) for e in check.expect)}"
            return Result(label, FAIL, f"{detail}\n{_tail(output)}", code, elapsed)
        if check.contains and check.contains not in output:
            return Result(label, FAIL,
                          f"output did not contain {check.contains!r}\n{_tail(output)}",
                          code, elapsed)
        if check.creates:
            target = Path(self.render(check.creates) or "")
            if not target.is_file() or target.stat().st_size == 0:
                return Result(label, FAIL, f"did not write {target}", code, elapsed)
        return Result(label, OK, "", code, elapsed)


def _tail(output: str, lines: int = 12) -> str:
    body = [ln for ln in output.strip().splitlines() if ln.strip()]
    return "\n".join(f"    | {ln}" for ln in body[-lines:])


def select_checks(checks: list[Check], quick: bool) -> list[Check]:
    """
    Which checks a mode runs.

    Quick keeps every route and drops the flag combinations, so the pytest
    wrapper still fails when a route breaks. Dropping routes instead would make
    the fast mode a different, weaker claim than the one this suite makes.
    """
    if not quick:
        return checks
    marked = [c_ for c_ in checks if c_.quick]
    if marked:
        return marked
    runnable = [c_ for c_ in checks if not c_.skip]
    return runnable[:1] or checks[:1]


def run_suite(sandbox: Sandbox, only: str = "", quick: bool = False,
              verbose: bool = False) -> tuple[list[RouteResult], list[str]]:
    runner = Runner(sandbox)
    declared = parser_routes()
    uncovered = [name for name in declared if name not in ROUTES]

    ordered = [name for name in ROUTE_ORDER if name in ROUTES]
    ordered += [name for name in ROUTES if name not in ordered]

    # `run` is what puts recordings in the index, so it happens whatever was
    # asked for. Reporting it even when it was filtered out is the honest thing:
    # it ran, and its result is part of what the other route's result means.
    wanted = [name for name in ordered if not only or name == only or name == "run"]

    results: list[RouteResult] = []
    for name in wanted:
        checks = select_checks(ROUTES[name], quick)
        outcomes = [runner.run_check(check) for check in checks]
        if name == "run":
            runner.refresh_fixture_ids()

        if any(r.status == FAIL for r in outcomes):
            status = FAIL
        elif all(r.status == SKIP for r in outcomes):
            status = SKIP
        else:
            status = OK
        note = "; ".join(r.detail.splitlines()[0] for r in outcomes if r.status == SKIP)
        results.append(RouteResult(name, status, outcomes, note))

        if verbose:
            for outcome in outcomes:
                print(f"  [{outcome.status:7s}] {outcome.label}  ({outcome.seconds:.1f}s)")
                if outcome.detail:
                    print(f"      {outcome.detail}")

    return results, uncovered


# =========================================================================
# Leaving nothing behind
# =========================================================================
def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """A recursive listing precise enough to catch a file being rewritten."""
    if not directory.exists():
        return {}
    listing: dict[str, tuple[int, int]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            listing[str(path.relative_to(directory))] = (stat.st_size, stat.st_mtime_ns)
    return listing


def describe_drift(before: dict[str, tuple[int, int]],
                   after: dict[str, tuple[int, int]]) -> list[str]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return (
        [f"added {name}" for name in added]
        + [f"removed {name}" for name in removed]
        + [f"modified {name}" for name in changed]
    )


# =========================================================================
# Reporting
# =========================================================================
def print_table(results: list[RouteResult], uncovered: list[str], drift: list[str],
                sandbox: Sandbox, elapsed: float, kept: Path | None) -> None:
    names = [r.route for r in results] + uncovered + ["route"]
    width = max(len(name) for name in names) + 2
    print()
    print(f"{'route'.ljust(width)}{'result':9s}{'checks':8s}note")
    print("-" * (width + 17 + 46))
    for result in results:
        checks = f"{result.passed}/{result.ran}" if result.ran else "-"
        print(f"{result.route.ljust(width)}{result.status:9s}{checks:8s}{result.note[:72]}")

    for name in uncovered:
        print(f"{name.ljust(width)}{FAIL:9s}{'-':8s}route not covered by scripts/smoke.py")

    for result in results:
        for check in result.checks:
            if check.status == FAIL:
                print(f"\nFAIL {result.route}: {check.label}")
                print(f"  {check.detail}")

    if drift:
        print(f"\nFAIL the repository's data/ changed during the run ({len(drift)} path(s)):")
        for line in drift[:20]:
            print(f"  {line}")

    failed = [r.route for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]
    print()
    print(f"{len(results)} route(s) in {elapsed:.0f}s, "
          f"{len(failed)} failed, {len(skipped)} skipped, "
          f"{len(uncovered)} uncovered, "
          f"{sandbox.server.completions} stub completion(s) served")
    if not drift:
        print(f"{REAL_DATA} is byte for byte what it was before this ran")
    if kept:
        print(f"temp project kept at {kept}")
    print("no network, no ffmpeg, no model weights, no API keys" if not failed
          else "RESULT: FAILED")


def as_json(results: list[RouteResult], uncovered: list[str], drift: list[str],
            sandbox: Sandbox, elapsed: float, kept: Path | None) -> str:
    return json.dumps({
        "ok": not any(r.status == FAIL for r in results) and not uncovered and not drift,
        "seconds": round(elapsed, 1),
        "stub_completions": sandbox.server.completions,
        "kept": str(kept) if kept else "",
        "uncovered": uncovered,
        "data_dir_drift": drift,
        "routes": [
            {
                "route": r.route,
                "status": r.status,
                "checks": [
                    {"command": c_.label, "status": c_.status, "exit_code": c_.exit_code,
                     "detail": c_.detail, "seconds": round(c_.seconds, 2)}
                    for c_ in r.checks
                ],
            }
            for r in results
        ],
    }, indent=2)


# =========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="smoke.py",
        description="Run every CLI route against a throwaway project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--only", default="", metavar="ROUTE",
                    help="run one route (the run route still executes; it is the fixture)")
    ap.add_argument("--quick", action="store_true",
                    help="one check per route rather than every flag combination")
    ap.add_argument("--list", action="store_true", dest="list_routes",
                    help="print what would run, and exit")
    ap.add_argument("--keep", action="store_true",
                    help="leave the temp project in place and print its path")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable result instead of the table")
    ap.add_argument("--verbose", action="store_true", help="print every command as it runs")
    ap.add_argument("--tmp-dir", default=None,
                    help="where to build the throwaway project (default: the system temp dir)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    declared = parser_routes()

    if args.list_routes:
        for name in declared:
            checks = ROUTES.get(name)
            if checks is None:
                print(f"{name:14s} NOT COVERED")
                continue
            for check in select_checks(checks, args.quick):
                marker = f"SKIPPED: {check.skip}" if check.skip else ""
                print(f"{name:14s} run.py {check.label} {marker}".rstrip())
        missing = [n for n in ROUTES if n not in declared]
        for name in missing:
            print(f"{name:14s} STALE: covered here but not in the parser")
        return 0

    if args.only and args.only not in ROUTES:
        print(f"unknown route '{args.only}'. Known: {', '.join(sorted(ROUTES))}")
        return 1

    before = snapshot(REAL_DATA)
    started = time.monotonic()
    root = Path(tempfile.mkdtemp(prefix="plaud-smoke-", dir=args.tmp_dir))
    sandbox = build_sandbox(root)
    try:
        results, uncovered = run_suite(sandbox, only=args.only, quick=args.quick,
                                       verbose=args.verbose)
    finally:
        sandbox.stop()
    elapsed = time.monotonic() - started

    # The repository's own data/ is the operator's real archive. A suite that
    # writes into it while claiming to be safe to run on a laptop has done the
    # one thing it promised not to, so this is a failure and not a warning.
    drift = describe_drift(before, snapshot(REAL_DATA))

    # A single route was asked for, so the coverage claim is not being made and
    # the uncovered list would be every other route.
    if args.only:
        uncovered = []

    kept = root if args.keep else None
    if args.as_json:
        print(as_json(results, uncovered, drift, sandbox, elapsed, kept))
    else:
        print_table(results, uncovered, drift, sandbox, elapsed, kept)

    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)

    failed = any(r.status == FAIL for r in results)
    return 1 if (failed or uncovered or drift) else 0


if __name__ == "__main__":
    raise SystemExit(main())
