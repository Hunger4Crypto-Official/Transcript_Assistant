# Changelog

## Unreleased — day-length recordings

### Episodes: a day becomes a rundown per profile

Wear the device from 8am to 6pm and the pipeline classified all of it as one
thing from a 14,000 character sample — about twenty minutes of speech — and
produced one summary of everything.

- Long recordings are cut into episodes, each routed on its own, and each
  profile's analysis is built only from the episodes that belong to it.
- Segmentation is deterministic and local: a silence gap, a change in who is
  talking, a drop in vocabulary overlap, and a maximum length. Every boundary
  records which signal produced it.
- A silence gap overrides the minimum-length rule. A two minute client call with
  five minutes of driving either side is its own conversation however short it
  is; merging it into the coaching session before it put two unrelated
  conversations into one profile's rundown.
- ADR-002 is unchanged: the strictest profile matched by any episode still
  governs the whole file.

### Offline is now an assertion

- `runtime.offline` refuses to load while any cloud provider is enabled.
- Models resolve against `runtime.models_dir`; missing offline, the error names
  the directory it wanted and the command that fills it.
- Diarization no longer needs a HuggingFace token once the weights are on disk.
- `scripts/fetch_models.py` collects models and wheels for the one trip across.
  `doctor --offline` audits it rather than assuming.

### Large files

A full working day is roughly 450MB as a 128kbps mp3, and encrypting the
original needed the plaintext and the ciphertext resident at once — so the
longest recording you own was the one that ran the machine out of memory.

- Originals are encrypted a chunk at a time. Measured: a 419MB file round-trips
  at 61MB peak RSS, against ~839MB for the one-shot path.
- Each chunk is bound by AAD to the recording id, its index, and whether it is
  the last one. That is what makes reordering, dropping a chunk, and truncation
  fail loudly instead of handing back a shorter recording.
- `verify` decrypts streamed artifacts to `os.devnull`, so checking a day of
  audio opens does not load it or leave a plaintext copy behind.
- `open --kind audio --out` streams out the same way.
- `ingest.max_file_mb` raised 512 → 2048. It is a disk budget now, not a memory
  one.
- Artifacts written before this format still open unchanged.

---

## Unreleased — full audit

A deliberate audit of the code against its own claims. Everything below shipped,
and none of it raised an exception — which is what made it worth finding.

### Content that should not have been in the clear

- **A recording that failed between transcription and the compliance gate wrote
  its full plaintext transcript into the unencrypted index, permanently.** The
  gate decides whether content may sit in the clear, and it had not run yet, so
  the flag still said "not encrypted". Reproduced with a family recording and a
  simulated router outage. `encrypt_at_rest` now defaults to True and is only
  relaxed by the gate once it knows the governing profile.
- **Encryption had two sources of truth.** `is_encrypted` came from sensitivity
  while persistence used the profile's `encrypt_at_rest`. A legal profile — high
  sensitivity, encryption off — made them disagree: artifacts written in
  plaintext while the index withheld them as encrypted, so the digest could
  never render that recording again. One field now, read by both.
- **The original recording was archived in the clear.** The most sensitive
  artifact there is — the actual voices — sat in `inbox/_processed/` while the
  transcript derived from it was encrypted beside it. Originals now go into the
  vault when the governing profile encrypts at rest. Retrieve with
  `run.py open <id> --kind audio --out file.mp3`.

### A search that did not look

- **`search --content` silently scanned only the 50 most recent recordings** and
  reported "nothing matching X was said". The CLI's `--limit`, which reads as
  "results to show", was being used as "recordings to open". It now scans
  everything by default, reports what it scanned, and exits non-zero when the
  answer is incomplete. `--scan-limit` bounds work explicitly and says so.

### Deletes that reached outside the tool

- **`forget` and `retention --execute` would unlink any path the index named.**
  Verified: a row pointing at an unrelated PDF outside the data directory
  deleted it. Both now refuse anything outside the configured directories, and
  log, audit, and report the refusal.

### Also fixed

- Audit retention was never applied; `audit_log_days` was parsed and read by
  nothing. Now swept, keeping the longest window any profile asks for, and
  visible in the dry run before you execute it.
- A file still being copied into the inbox could be ingested half-written, and
  the partial file's content hash is what dedupe remembers forever.
  `ingest.settle_seconds` skips it; a run that processed nothing says why.
- Two processes racing on the same file (a `watch` loop and a manual `run`) hit
  the UNIQUE constraint and reported the recording as failed. It is a duplicate.
- An encrypted text import produced an orphan vault file nothing pointed at and
  no sweep would expire. Originals are indexed whatever their kind.
- `retention --execute` skipped audit-only plans because it tested `plan.items`.
- Filename search treated `%` and `_` as LIKE wildcards, so `100% done.txt`
  matched everything.
- `export` rendered list fields as raw JSON; it now uses the digest's renderer.
- `export --transcripts` necessarily includes the material `suppress_fields`
  exists to withhold. It now says so, naming the fields, before writing.
- `review` crashed on a timezone-naive audit timestamp.
- Three copies of the same timestamp formatter, now one.

### Testing

128 → 143. `tests/test_audit_regressions.py` reproduces each defect above.

---

## Unreleased — getting things back out

The pipeline could put recordings in. Little could get them out again.

### Content search

- `run.py search "own occupation" --content` searches what was actually said,
  decrypting the vault where it has to, and prints the timestamp and speaker of
  every hit. `--context N` widens each hit to the surrounding speech.
- Search previously only matched filenames, which for a Plaud export means
  matching `REC0042.wav`.
- **It reports what it could not open and exits non-zero.** Returning fewer
  results because a file would not decrypt is the worst failure a search over
  your own archive can have: you conclude the phrase was never said.

### Verify

- `run.py verify` opens every artifact the index points at. Catches missing
  files, silent corruption, and a wrong passphrase.
- A locked vault reports encrypted artifacts as *unchecked*, not healthy. It
  will not claim a clean bill it could not confirm.
- Lists files on disk the index does not know about. Never deletes anything.

### Forget

- `run.py forget <id>` deletes one recording completely: vault artifacts,
  outbox files, archived original, quarantine folder, index row. Shows the exact
  file list and requires typing `FORGET`.
- The audit entry survives, and is written before anything is removed. An audit
  trail that forgets deletions is not a trail.
- Previously the only way to remove a single recording was hand-editing SQLite.

### Export

- `run.py export` builds a document meant to leave the machine: redaction
  applied, suppressed fields still never included, personal profiles refused
  unless forced. `--transcripts` exports redacted transcripts instead of the
  analysis; `--format html` for something printable.
- The redaction footer prints even when nothing matched, because "no pattern
  fired" is not the same as "nothing sensitive is in here".

### Running it without remembering to

- `run.py watch --interval 300` polls the inbox until you stop it.
- README documents cron scheduling, including the failure mode where a cron job
  without `PLAUD_BRIDGE_PASSPHRASE` transcribes and then refuses to write.
- `make verify`, `make review`, `make week-html`.

### Internal

- `archive.py` is now the single path for reading stored content back. The rule
  about where a recording's words live — payload when unencrypted, vault only
  when encrypted — was about to be duplicated into four readers.

---

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
