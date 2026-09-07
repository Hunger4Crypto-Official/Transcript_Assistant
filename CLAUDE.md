# Plaud Bridge — working notes for a new session

Read this before touching anything. It is the context a fresh chat does not
otherwise have: what this is, what has been settled, and what is actually true
about its state.

## What this is

A local, privacy-critical pipeline that turns recordings into digests, briefs,
and answers. Owner: Sasson (hunger4crypto@gmail.com). **Proprietary — see
LICENSE. All rights reserved, owner-only.** Nothing here is open source and
nothing should be published, hosted, or shared to a third party.

The whole design premise: **audio and transcripts never leave the machine
unless a work profile explicitly allows it.** Family and spousal recordings are
hard-locked to local processing and encrypted at rest, regardless of any
setting a person clicks.

## Non-negotiable invariants

Do not "simplify" any of these. Each is load-bearing, each has tests, and each
was argued for in an ADR or a red-team pass:

- **Strictest profile governs.** When a recording matches several profiles, the
  most restrictive locality and encryption setting wins (ADR-002).
- **A refusal is never released by a click.** If a party objected to being
  recorded, the app has no release button — only a deliberate CLI path.
- **Plaintext never touches disk.** The media player streams originals out of
  the vault decrypted chunk by chunk. No temp files, no decrypt-then-serve
  staging, ever. See `media.py`'s module docstring.
- **Nothing is invented.** Quotes shown anywhere are verified verbatim against
  the source with the shared `quote_is_present` helper. A model's fabrication is
  dropped, counted, and reported — never rendered.
- **A label is attribution, not identity.** Speaker names carry
  `voice_verified`; placeholders are bucketed as "(unidentified speakers)"
  rather than presented as people.
- **`forget` reaches everything.** Drafts, answers, memory, follow-ups,
  quarantine. A derived cache that survives `forget` is a leak — which is why
  `insights.py` deliberately stores nothing.
- **Honesty over completeness.** A search that could not open three recordings
  says so. Exit 2 means "answered but incomplete" and is not a crash.

## The gates — run all three before claiming anything works

```bash
.venv/bin/python -m pytest tests/ -q        # append: ; echo "EXIT=$?"  (unpiped!)
.venv/bin/python -m ruff check .
.venv/bin/python scripts/smoke.py --quick   # drives the real CLI in subprocesses
make coverage PYTHON=.venv/bin/python       # the CI floor, line+branch (~15 min)
```

`make` targets run bare `python`; without `PYTHON=.venv/bin/python` on a box
where that is the system interpreter, `make coverage` dies on an unknown
`--cov` flag before running a single test. Ruff lints Python only -- passing
it the Makefile or a YAML file produces a wall of "syntax errors" that mean
nothing.

Piping pytest through `tail` reports the pipe's exit code, not pytest's. Always
echo `$?` from the unpiped command.

Adding a CLI route means updating four places or the parity tests fail, by
design: `cli.py`'s parser + docstring, `run.py`'s docstring, `scripts/smoke.py`
(ROUTES + ROUTE_ORDER), and `tests/test_cli_routes.py` (COVERED + READ_ONLY).

Tests are named as sentences describing the behavior they pin. Fixtures live in
`tests/_fixtures.py` (`build_sandbox`, `drop`, `StubLLM`, `CLIENT_CALL`,
`FAMILY_DINNER`). When fixing a bug, verify the fix by mutation: break the fix
deliberately, confirm the test fails, restore. Restore with a `cp` backup —
`git checkout <file>` has silently destroyed uncommitted work here twice.

## Measured state (not aspirational)

- **Ruff clean. Every smoke route passes.** Test count and coverage move; the
  numbers below are the last measurement, and `make coverage` re-measures.
- **Coverage is 85% line+branch** (87% line alone), not 100% — measured
  2026-09-06: 1,228 of 9,336 statements and 446 of 3,046 branches never run.
  CI enforces a floor of 85 (`COVERAGE_FLOOR` in the Makefile) that only
  moves up. Raise it when the measurement rises; never lower it.
  - **`compliance/gate.py` is at 100% line and branch**, mutation-verified
    (`tests/test_compliance_gate_edges.py`). It was the priority gap; closed.
  - Very little is genuinely blocked. `http_util`, the ASR/LLM providers, and
    the diarization engine can all be driven by loopback stub servers or an
    injected fake module — "needs network" was an excuse, not a fact. Only
    real model weights, real audio through Whisper, and the Windows updater
    are unreachable here.
  - Largest remaining gaps at last measurement: `cli.py` (256 uncovered),
    `desktop/server.py` (79), `voiceprint.py` (76), `memory.py` (72),
    `http_util.py` (65), `archive.py` (63), `diarize/engine.py` (53).
- Coverage measures code executed, not behavior asserted. A test that touches
  a line without asserting its effect does not count here — pin the behavior.

## Settled decisions — do not relitigate

- **No internet hosting.** Not Vercel, not anything. The architecture is
  local-first and the data is the reason.
- **Rejected by design, with reasons on record:** plaintext follow-up state
  (M3), imported-VTT consent spoofability (inherent to accepting exports),
  ReDoS via self-authored config (M9, input now bounded).
- The `MembersOnlyOfficial` / `$MemO` repository is a **different project** and
  is off-limits unless explicitly asked for.
- Type checking as a CI gate was deferred — owner's call, not yet made.

## Environment

- Remote container, recycled without warning. Rebuild: `python3 -m venv .venv &&
  .venv/bin/pip install -e ".[dev]"`.
- Network policy blocks huggingface.co, groq, azure. pypi and github work.
- ffmpeg installs via apt. faster-whisper installs via pip.
- Branch: `claude/build-out-feature-udg7jl`. Push with
  `git push -u origin claude/build-out-feature-udg7jl`, retrying 2/4/8/16s on
  network failure. Never push elsewhere.

## Known open items that need the owner, not code

1. **First real Windows build.** Trigger *Build Windows app* in the repo's
   Actions tab, then debug the log. Never been run.
2. **First real audio run.** Nothing has ever been processed from actual mp3 —
   only text fixtures. Needs either a local machine or this environment's
   network policy opened to `huggingface.co` + `cdn-lfs.huggingface.co` for the
   Whisper weights.
3. Optional: a Tailscale/VPN guide for using Phone mode away from home.

## Tone for reports

Say what is measured, not what is hoped. "All tests pass" and "everything is
tested" are different claims. If something is blocked, say which part and why,
finish everything else, and name what was left out.
