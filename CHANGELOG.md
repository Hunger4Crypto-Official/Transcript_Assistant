# Changelog

## Unreleased — the second brain: brief, people, insights, and the player

Four engines that turn an archive of recordings into something no note-taker
ships, each headless-first (a CLI route wired into the smoke suite) and then
surfaced in the app:

- **`brief`** — the week synthesised across every recording: what moved, which
  promises are aging, who is waiting on you, what to do next. Built in two
  layers with a hard line between them: a deterministic skeleton read entirely
  off disk (renders completely with no model at all, labelled **assembled**),
  and a narrative layer whose every quoted receipt is verified verbatim with
  the extractor's own `quote_is_present` — a fabricated quote or an un-sent
  recording id is dropped, counted, and reported, never rendered. Locality and
  redaction verdicts come from `ask`'s own functions, so the brief can never
  reach a softer privacy conclusion than a question would.
- **`people`** — one page per person, assembled from everywhere they were
  heard: minutes, topics, verified quotes, commitments in both directions, and
  every appearance. Two honesty rules are baked into the data: a speaker label
  is attribution, not verified identity (`voice_verified` says which, and
  placeholder labels are bucketed as "(unidentified speakers)" rather than
  invented into a person), and nothing is shown that was not already verified
  verbatim upstream. Personal recordings stay off the roster unless asked.
- **`insights`** — how you actually talk, measured rather than remembered:
  talk share, pace, question rate, longest monologue, silence share, and
  overlap-based `interruptions_approx` (named so nobody quotes it without the
  caveat), per speaker, per recording, and as a 30-day-vs-prior trend. Pure
  arithmetic over the stored segments — no model, and nothing stored, so
  `forget` has nothing new to chase.
- **The moment player** — click a recording in the Library and hear it. The
  original streams from the vault decrypted chunk by chunk straight into the
  response; **plaintext never touches disk** — no temp files, no
  decrypt-then-serve staging. Range requests are honoured exactly on plaintext
  originals (scrubbing works) and refused on encrypted ones rather than staging
  a decrypted copy to satisfy the seek — players buffer forward instead. The
  synced transcript highlights the line under the playhead and clicking a line
  jumps the audio. A tampered or truncated vault stream still dies loudly
  mid-body; silent truncation stays impossible.

The app grew **Brief**, **People**, and **Insights** tabs on the same
token-guarded loopback server, all escaped before touching the DOM, all
invalidated when a processing run finishes.

## Earlier unreleased — the clickable app, the charts, and the safety net

### A window instead of a terminal

Double-click and use it from a browser: `desktop_app.py` (and the packaged
Windows build under `packaging/`) opens a local app that is a thin skin over
the same pipeline; the locality locks, consent gate, and vault keep their
single implementation. Seven tabs:

- **Process** — passphrase, the Offline/free-cloud-key brain switch, readiness
  checks, drop recordings, watch the run. A run that holds recordings points
  you at the Held tab.
- **Library** — every recording with profile, minutes, and encrypted/held
  badges, plus digest controls: a time window and an include-personal toggle
  (with the over-someone's-shoulder warning), opening the charts digest.
- **Search** — what was actually said, with the honesty line intact: it says
  how many recordings were searched and how many could not be opened, so a
  partial search never presents as a complete one.
- **Ask** — a question answered with citations; with no model reachable it
  says plainly that it is showing ranked excerpts, never a fabricated answer.
- **Follow-ups** — the commitments worklist, oldest first, with Done/Drop/
  Reopen persisting through the engine's own status rules.
- **Held** — quarantine triage with the reason classed and badged. Releasing
  is a human affirmation; a recording where someone *objected* has no release
  button at all and says why — a refusal is not a click. Forget requires
  typing the word and routes through the real `forget`.
- **Tools** — one-click encrypted backup, and the app's version.

Everything the page renders is escaped before it touches the DOM — transcript
text is untrusted content even in your own browser.

**Phone mode** (`--phone`, or `PLAUD_BRIDGE_PHONE=1`): the app also answers on
your Wi-Fi address, so a phone on the same network opens the same tabs and
"Add to Home Screen" installs it like an app — no cloud involved. Opt-in every
launch, never a default. The token stays mandatory on every request, and over
the network that includes the page itself, so nobody on the Wi-Fi can be handed
the key by asking; a served manifest makes the install real. Home network only:
the link carries the session's key and Wi-Fi traffic is unencrypted.

- Loopback-only, token-guarded, Host-checked. Another machine — or a hostile
  website you happen to have open — cannot drive it.
- The Offline brain diagnoses itself: it probes the local model server and
  tells you the one command to run — "install from ollama.com" vs
  `ollama pull <model>` vs ready. Choosing Offline enables `llm.local` in
  memory; the file on disk is never rewritten.
- `.github/workflows/build-windows.yml` builds the double-clickable
  `PlaudBridge.exe` on a Windows runner; see `packaging/README.md`.
- The app keeps itself current: it checks GitHub Releases on launch and shows
  an "Update available" banner; one click downloads the new build, verifies it
  against the SHA-256 the CI published beside it, and swaps itself out.
  Deliberately not silent -- an app that replaces its own executable in the
  background is a supply-chain attack with a release schedule. A release
  without its checksum is refused. `PLAUD_BRIDGE_NO_UPDATE_CHECK=1` turns the
  check off; a private repo needs `GITHUB_TOKEN` for the check to see releases.

### The digest grew charts

`digest --format html` now opens with an "In Charts" section: minutes per
section, minutes per day across the window (weeks when the window is long),
and API spend per section. Pure inline SVG — no scripts, no network, prints
cleanly, light and dark. Charts are computed from the same sections the text
was rendered from, so an excluded personal profile cannot appear in a chart by
construction.

### One file that brings the archive back

`backup` writes the vault, index, outbox, quarantine, and your tuned config
into a single file encrypted with the vault's own streaming cipher — safe on
an external drive or in a cloud folder, and refused outright without a
passphrase. `restore` decrypts, unpacks, and verifies everything in a staging
directory first; a wrong passphrase or tampered file changes nothing. The
index is snapshotted through SQLite's backup API so a live database cannot
restore as corruption.

### Triage the quarantine at scale

`quarantine` lists everything held, with the reason distilled: explicit
refusal, no announcement found, or a static consent gate. `--release-all`
demands its own typed confirmation and never includes a refusal — those stay
one-at-a-time on purpose. `--forget-all` routes through the real `forget`,
inheriting its locked-vault refusal and derived-store purging. A run that
quarantines now points you here.

### Measure before you migrate

`python scripts/bench.py recording.mp3` times ffmpeg and local transcription
on one real file and projects your backlog: the realtime factor on *this*
machine is the number that decides local-versus-cloud, and specs do not know
it.

## Unreleased — names, answers, memory, and follow-through

### Ask the archive a question

`ask "what did I promise the Hendersons?"` answers from what was actually said.
Retrieval runs first and is deterministic; only the excerpts it found are ever
shown to a model.

- Every citation is validated against the bundle that was actually sent. One
  naming a recording that was never sent is dropped and reported by id. See
  ADR-024, and the mutation test that deletes the check to prove it matters.
- With no model configured the command still works, returning ranked excerpts
  and saying plainly that they are search output rather than an answer.
- Exits 2 when the answer is incomplete; 0 when "nothing matched" is the
  complete and honest answer.
- The strictest profile in the bundle decides whether the call may leave the
  machine. Personal profiles stay out unless asked for.

### It carries what it knows forward

Every recording used to be analysed as though it were the first one ever seen.

- Each profile keeps a ledger of people, open commitments, and recurring topics,
  and the next analysis for that profile is made knowing them.
- The ledger is derived and rebuildable — `memory --rebuild` replays the archive
  — so it can never quietly become a second uncontrolled copy of it. ADR-026.
- Profile isolation is enforced by the cipher, not by convention: each ledger is
  encrypted under its own AAD.
- Entries decay. A commitment closes only when a later recording says it was
  done, because guessing closure from repetition would be inventing. ADR-027.
- `forget` now reaches memory, or its promise would be false.

### Follow-ups, drafted and never sent

- Commitments are collected across recordings, deduplicated by content so the
  same promise in three conversations is one item, and aged so the oldest debt
  sorts first.
- `--done` persists, so a closed item stops resurfacing.
- `--draft` writes into the outbox. There is no send path in the code at all —
  no SMTP, no mail API, no configuration for one. ADR-025.
- Drafts are redacted unconditionally, diverging from `export` on purpose: a
  draft is outbound by definition.

### The inbox takes what note takers actually produce

"Works with your recorder" and "works with everything that records" are
different products, and the difference was an extension list and a parser.

- WebVTT is now a first-class transcript format — Zoom, Teams, Fireflies,
  tl;dv, and YouTube all export it. Header, NOTE, STYLE, and cue-identifier
  handling included, hourless timestamps and comma milliseconds tolerated,
  markup stripped.
- **Teams voice tags become named speakers.** `<v Marcus Reed>` is the platform
  stating who spoke from its own per-participant channels, so it is treated as
  authoritative and flows into the stored transcript unchanged — named speakers
  with no diarization, no enrollment, and no model.
- The audio list grew from five extensions to fifteen: phone memos (`.m4a`
  `.m4b`), WhatsApp and Telegram voice notes (`.opus` `.oga` `.amr`), and
  meeting recordings (`.mp4` `.mov` `.webm`, audio extracted). ffmpeg already
  normalised any container; the extension list was the only gate.
- Unsupported files are still refused by name, never silently ignored.

### The brain, brought up to date

The analysis model was two generations stale, and upgrading it would have failed
on the first call.

- `claude-opus-5` replaces `claude-sonnet-4-6`, with the rate table updated to
  match. The pricing tests now read the rates from config instead of restating
  them, so they stop failing the day a price moves.
- **No sampling parameter is sent to Anthropic.** `temperature` was pinned to
  0.0 for determinism it never provided; the current models reject it outright,
  so it was a 400 waiting for the next model bump rather than a harmless
  leftover.
- **The system prompt is cached.** A profile's instructions and schema are
  identical across every recording and every episode, and were being paid for at
  full price every time. ADR-030.
- `max_tokens` raised, because thinking is on by default on this generation and
  shares that ceiling with the response — the old budget truncated mid-JSON.
- `effort` is now a config key, and it is the cost lever that replaced the token
  budget.

**The free Groq key is untouched** — its LLM and its Whisper ASR both keep
working exactly as before. The sampling parameter was removed on one vendor's
models, not on every endpoint that speaks the same wire format, and a test now
fails if anyone "helpfully" strips Groq's too.

### A quote is findable, or it is dropped

`ask` validated its citations. The extractor, which produces the promises that
flow into memory and the follow-up worklist, was on the honour system — the
prompt said "the speaker's exact words" and nothing checked.

Now every `quote` field is checked against the text the model was actually shown
— redacted, when compliance redacted it, since that is all the model could have
quoted. Case, punctuation, and whitespace are forgiven; different words are not.
Dropped quotes are counted on the analysis and named in the log. ADR-029.

### A transcript it was guessing at now says so

Speech recognition does not decline. Given music, a restaurant, or a device in a
pocket it returns fluent English that nobody said -- and everything here treated
the transcript as fact, so an invented sentence became a promise the worklist
put in front of you.

Every segment has carried a confidence score since the first version and nothing
read it. Now the pipeline does, along with the recogniser's own estimate that a
span held no speech at all, which is the signature that catches the confident
inventions the log probability alone misses.

- A bad transcript is marked, audited, and announced in the digest **above** the
  analysis, because everything below that line came out of it.
- The extraction prompt is told to prefer empty fields, last, beside the
  instruction -- a caveat given as background gets noted and ignored.
- Weighted by duration, so four minutes of invented music cannot hide behind a
  crowd of real two-second replies.
- Imported text reports *unknown*, not clean. It has no scores, and calling it
  clean would claim a check that never ran.
- Nothing is deleted or refused. A quiet conversation in a car scores badly and
  is still the conversation you wanted. See ADR-028.

The thresholds are guesses. Check them against your own microphone.

### Two things the new features quietly broke, found afterwards

- `verify` reported voiceprints, saved answers, and drafts as "files on disk
  that the index does not know about", next to advice about rebuilt databases
  that could not apply to them. Three features had started writing into the
  vault and the outbox on purpose. They are now counted and named as
  non-artifacts, and a real orphan still stands out beside them -- the fix is
  not "stop looking in the vault".
- `ask` and LLM-phrased drafts spent money that `status` could not see, which
  contradicts ADR-014. Spend that belongs to no recording now has a table of its
  own, and `status` breaks the total down by where it went.

### A smoke suite that drives the real command line

`scripts/smoke.py` stands up a throwaway project, serves its own model on
loopback, and runs every route as real subprocesses — no ffmpeg, no weights, no
keys, and it verifies your own `data/` is byte for byte unchanged afterwards.

The route list is read from the parser at runtime, so a new subcommand with no
coverage fails the run instead of quietly shrinking what "every route" means.

**Two defects it found, both fixed:**

- A quarantined recording made `search --content`, `export`, and `ask` exit 2
  forever, advising a passphrase fix that could not help. The gate stops those
  recordings before anything is written, and the archive was reporting "could
  not open" for content that had deliberately never been stored.
- `run --force` on a still-quarantined file minted a new recording id, wrote a
  second quarantine folder, then failed to index it on the UNIQUE content hash
  and blamed a concurrent process. The surviving folder belonged to a recording
  no index knew about.

## Previously — named speakers

### Speakers have names now, or they stay numbered

Diarization could tell three people apart but had never heard any of them
before, so the best it could write was `Speaker 2`.

- `speakers enroll "Marcus" --audio clip.wav` learns a voice. `--start/--end`
  trim to a clean stretch; a second clip from a different room improves it.
- `speakers identify <audio>` prints every cluster, its length, and its
  similarity to everyone enrolled, and writes nothing. It is how you pick a
  threshold for your own microphone instead of trusting a default.
- `speakers list` and `speakers forget` complete the set. `doctor` reports the
  embedding model and who is enrolled.
- Enrolling yourself beats `assume_owner_is_dominant_speaker`, which is a guess
  that is usually right and occasionally embarrassing.

### It would rather say nothing than guess

A name is believed in a way `Speaker 2` is not, so two guards stand between a
similarity score and a transcript: an absolute threshold, and a margin over the
runner-up. Two brothers at 0.61 and 0.60 is a tie, not an identification, and
both stay numbered. A person is used at most once per recording. See ADR-022.

### Voiceprints are encrypted or they are not stored

Enrollment requires a working vault passphrase, with no plaintext fallback and
no flag to ask for one. A voiceprint is biometric data about people who did not
install this software; a plaintext copy of it is a biometric database in a user
directory. See ADR-023.

Nothing is uploaded — the embedding model runs locally like everything else.
`scripts/fetch_models.py --embedding` collects it for an air-gapped machine.

## Previously — day-length recordings

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

## Previously — full audit

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

## Previously — getting things back out

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

## Previously — voice, templates, and the review cadence

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

## Previously — unpacking and enforcement

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
