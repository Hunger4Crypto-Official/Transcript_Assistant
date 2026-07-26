# Changelog

## Unreleased — voice, templates, and the review cadence

### Voice

The digest's wording was hardcoded prose scattered through the renderer.
Changing "Needs You" to something that sounded like you was a source edit.

- Every user-facing string now comes from a voice pack in `config/voice/`.
  Three ship: `plain` (neutral, the default), `brief` (short and scannable, for
  a phone between appointments), and `warm` (written like a person assembled
  it, for the personal profiles).
- `voice.overrides` in `pipeline.yaml` changes individual strings without
  copying a whole pack.
- Profiles carry their own voice: `digest.intro` opens a section,
  `digest.empty` is what it says when the window turned up nothing, and
  `extraction.persona` sets the register for that profile's analysis.
- `analysis.house_style` sets one register across all five profiles so they do
  not drift apart. Prompt layering is house style, then persona, then the
  profile's `system_prompt` — constraints last, nearest the task, where nothing
  configured above can dilute them.
- Structure stays in code. A pack supplies words and cannot reorder or re-emit
  sections, because the renderer is where suppressed fields, personal-profile
  exclusion, and on-demand decryption are enforced. There is a test that an
  override cannot talk the digest into printing a suppressed field.
- Nothing can fail to render: a partial pack is valid, a missing or corrupt one
  falls back with a warning, and an unknown placeholder renders empty.

### Templates

- `config/profiles/_TEMPLATE.yaml`, documenting every key inline, including the
  ones that are easy to get wrong: what sensitivity actually drives, why cloud
  ASR is opt-in per file, and what `requires_human_attention` does.
- `run.py new-profile <id> --name ... --heading ...` scaffolds from it.
- Files starting with an underscore are skipped by the loader, so the template
  never becomes a profile called TEMPLATE.

### The review cadence

COMPLIANCE.md section 9 asks you to read certain things weekly, monthly,
quarterly, and annually. Nothing implemented it.

- `run.py review` assembles all four tiers and reports what is actually due.
- `reaffirm_every_days` now does something. `father.yaml` promised a prompt "on
  first run each month" and the value was parsed and never read; lapsed standing
  consent is now surfaced, and `run.py review --reaffirm <profile>` records the
  reaffirmation to the audit log as a human action.
- Unfiled recordings are harvested for the keywords that would have routed them.
- Expired artifacts are reported. `review` never deletes anything.

### HTML digests

- `run.py digest --format html` writes a self-contained page: no scripts, no
  fonts, no external stylesheet, so opening a digest containing a client's
  health disclosures makes no network request. Light and dark, and it prints.
- Rendered from the same markdown, so the two formats cannot drift into saying
  different things.

### Fixed

- The reaffirmation lookup kept the oldest entry per profile rather than the
  newest, which would have reported a profile as overdue forever.

---

## Unreleased — unpacking and enforcement

The repository previously held the project as a `.tar.gz` with a few files
unpacked beside it. This release makes it a working repository, then fixes what
auditing the code against its own documentation turned up.

### Privacy guarantees that were documented but not enforced

Four of these. Each one read as reasonable in isolation, and each one broke a
promise the README states in plain language. All four are now pinned by
`tests/test_privacy_guarantees.py`.

- **Analysis ignored the compliance gate's verdict.** `force_local_processing`
  was computed and read by nothing. Each analysis recomputed locality from its
  own profile, so a recording matching both Husband and Sales Trainer kept the
  Husband analysis local and then sent the identical marital transcript to a
  cloud provider for the Sales Trainer one. The governing profile now decides
  for the whole recording, and per-profile policy can only make it stricter.
- **Cloud transcription was the default.** ASR locality came from a filename
  keyword check that defaulted to cloud, so a Plaud export named `REC0042.wav`
  went to a third-party transcription service regardless of what was on it.
  Cloud ASR is now opt-in per file: the filename must name a profile that allows
  it, and no locked profile may match.
- **Routing went to the cloud on a coincidence.** The routing call ships up to
  14,000 characters of transcript and was kept local only when a locked
  profile's keyword happened to appear. It is now local unless every routable
  profile permits a cloud LLM.
- **The index kept a plaintext copy of encrypted content.** `data/bridge.db`
  stored the verbatim transcript and the extracted quotes for recordings whose
  artifacts were encrypted, and retention never cleared it — after the sweep the
  encrypted copy was gone and the plaintext one remained. The index now holds
  metadata only for those recordings; the digest decrypts on demand.

### Consent

- A refusal is no longer read as consent. "I really don't want this being
  recorded" matched the announcement patterns, and a sympathetic reply from the
  next speaker completed it. Objections now veto the window outright.
- The announcement must come from `diarization.owner_label` when that speaker
  can be identified. The other party stating that *they* are recording is not
  you obtaining their consent.

### Spend

- LLM calls are priced from reported token usage at rates in `pipeline.yaml`.
  Previously no provider set `cost_usd`, so `cost.halt_usd_per_run` was an
  ASR-only ceiling and `status` under-reported.
- Routing cost is carried out of the router instead of discarded.
- Quarantined and failed recordings count toward the run total. They still paid
  for the calls that got them there.
- ASR cost survives a provider failover instead of being reset.

### Logging

- `logging.redact_content` did nothing: it keyed off an `extra={"content": True}`
  marker that no caller passed. It now applies `compliance.redact_patterns` to
  every line.
- Tracebacks are redacted. They were not touched at all, which is where it
  mattered most — provider errors fold the raw API response body and the full
  URL into the exception.

### Transcript import and stitching

Quiet failures: nothing raised, the run reported success, words were missing.

- SRT cues no longer lose numeric lines. The filter meant to strip the cue index
  stripped every numeric line, deleting policy numbers, dollar figures, and
  years — including `1035`, which the glossary exists to protect.
- `Well: I told him no.` no longer loses the word "Well" to a speaker called
  Well. A label is accepted when it repeats or reads like a name.
- The synthesised plain-text timeline is monotonic. Segments could start before
  the previous one ended, which rendered out of order and handed the consent
  detector an incoherent window.
- Chunk stitching anchors the duplicate window to the chunk start rather than
  the first surviving segment. With VAD trimming leading silence the window
  extended past the real overlap and deleted genuine speech.
- Non-Latin transcripts survive stitching. The normaliser stripped everything
  outside ASCII, so any two CJK or Cyrillic segments compared as identical and
  the second was dropped.
- Two speakers saying the same word a second apart stay two segments.
- Glossary corrections clean up the trailing stop ("The elimination period. is
  ninety days") and no longer count identity rules, which produced audit lines
  reading "3 corrections across 0 segments".

### Retention

- `execute()` refuses a dry-run plan instead of deleting from it.
- Expiry runs from when the recording was made, not when it was processed.
  Importing an old backlog no longer resets the clock on all of it.
- Original audio is indexed and swept. `raw_audio_days` was read by no code, so
  audio was the one artifact that never expired.

### Added

- `pyproject.toml`, installable, with a `plaud-bridge` console script.
- GitHub Actions CI: tests on Python 3.10–3.13, lint, and a separate
  `compliance guards` job so a red X reads as "a privacy guarantee broke".
- `run.py audit` — COMPLIANCE.md section 8 committed to an audit trail with no
  way to read it.
- 81 tests, up from 44. No network, no API keys.

### Other fixes

- `ffprobe` returning `"duration": null` produces an actionable error instead of
  an uncaught `TypeError`.
- Archived originals are prefixed with the recording id. The inbox is scanned
  recursively and the archive is flat, so two files with the same name silently
  overwrote each other.
- A routing reply that parses but scores nothing falls back to keywords instead
  of collapsing every confidence and filing the recording under `unfiled` — a
  profile with weaker retention and none of the family prompt's constraints.
- Bad `chunk_overlap_seconds` / `max_chunk_mb` combinations fail at startup
  rather than mid-run.
- `urllib` retry loops raise a real error instead of a bare `assert`, which
  vanishes under `python -O`.
