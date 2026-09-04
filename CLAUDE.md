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
```

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

- **829 tests pass. 32 smoke routes pass. Ruff clean.**
- **Coverage is 87%**, not 100% — 1,240 of 9,336 statements never execute.
  - ~250 are genuinely blocked in a sandbox: real Whisper weights, cloud HTTP
    (`http_util` 24%), pyannote diarization (59%), the Windows updater.
  - ~640 are simply untested and reachable. Largest: `cli.py` (256 uncovered),
    `desktop/server.py` (79), `voiceprint.py` (76), `memory.py` (72).
  - **`compliance/gate.py` is at 82%** — 14 statements in the consent-decision
    path have never run. This is the highest-priority gap; it was the next
    thing being worked on.
- Coverage measures lines executed, not behavior asserted. Treat 87% as an
  upper bound on what is genuinely pinned. Branch coverage was never measured.

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
