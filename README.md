# Plaud Bridge

Takes audio or transcripts from any recorder or note-taking app — Plaud, a
phone's voice memos, Zoom, Teams, WhatsApp voice notes, or typed markdown — and
turns them into transcripts, per-profile analysis, and a digest you can read or
filter. Runs on your machine. Nothing about a family conversation ever leaves
it.

The name stays: it was born to serve a Plaud recorder, and renaming a tool
because it learned to eat more formats would churn every config and environment
variable for a sentimentality nobody asked for.

Four profiles ship configured: **Insurance Agent**, **Sales Trainer**,
**Father**, **Husband**, plus an **Unfiled** catch-all.

---

## What it actually does

```
inbox/  ->  normalise  ->  chunk  ->  transcribe  ->  diarize  ->  glossary fix
        ->  route (multi-label)  ->  COMPLIANCE GATE  ->  analyse per profile
        ->  encrypt + index  ->  digest
```

Two design choices drive everything else:

**One recording can belong to several profiles.** A dinner where a client calls
is Husband and Insurance Agent at the same time. Forcing one label would lose
either the client follow-up or file a private conversation under work.

**The strictest matched profile governs the whole file.** One private sentence
in a business meeting locks the entire recording down. The alternative is
deciding mid-file that half a conversation is safe to ship to a third party,
which is not a decision software should make.

---

## Install

```bash
git clone <your-repo> plaud-bridge && cd plaud-bridge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ffmpeg is required
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu

cp .env.example .env         # fill in PLAUD_BRIDGE_PASSPHRASE at minimum
set -a; source .env; set +a

python run.py doctor
```

`doctor` tells you exactly what is missing and what will break as a result.
Fix every `FAIL` before your first real run.

### See it working before you record anything

```bash
python run.py demo --process     # fictional samples, through the real pipeline
python run.py brief              # then look around
python run.py people
python run.py insights
python run.py digest --format html --out data/outbox/demo.html
```

The samples are labelled fiction in their own first line, and they are chosen
to show several states at once: a consented client call (encrypted), a
coaching session (plaintext at rest), a family conversation (forced local and
kept out of shared digests), and one call with no consent exchange — which the
gate holds, so you watch it work rather than take its word for it.

`python run.py demo --clean` removes any samples still waiting in the inbox;
anything already processed comes out with `forget <id>` like any recording.

---

## Daily use

```bash
# 1. In the Plaud app: open a recording, export, choose Audio (WAV or MP3).
#    Drop the file into data/inbox/
# 2.
python run.py run

# 3. Read it
python run.py digest                        # last 7 days, work profiles only
python run.py digest --profile husband      # one profile
python run.py digest --profile father --days 30
python run.py digest --include-personal     # everything in one document
python run.py digest --days 7 --out data/outbox/week.md
```

Other commands:

| Command | What it does |
|---|---|
| `run.py doctor` | Preflight every dependency, key, and profile |
| `run.py demo --process` | **Fill it with labelled samples and process them** |
| `run.py app` | The local app, from the terminal |
| `run.py app --probe` | Start it, check it answers, exit — a headless self-test |
| `run.py status` | Index summary: counts, hours, spend |
| `run.py watch` | Process the inbox on an interval until you stop it |
| `run.py search "elimination period"` | Find recordings by filename |
| `run.py search "own occupation" --content` | **Search what was actually said** |
| `run.py ask "what did I promise the Hendersons?"` | **Answer a question, with citations** |
| `run.py verify` | Confirm every artifact still exists and still decrypts |
| `run.py export --days 30` | Build a redacted document for someone else |
| `run.py forget <id>` | Permanently delete one recording |
| `run.py memory` | **What it has learned across recordings** |
| `run.py brief` | **The week in one memo, every claim receipted** |
| `run.py people` | **One page per person, from everywhere they were heard** |
| `run.py people --name "Marcus"` | One person's whole dossier |
| `run.py insights` | **How you talk: share, pace, questions, monologues** |
| `run.py insights --recording <id>` | One recording's per-speaker breakdown |
| `run.py memory --brief` | The briefing injected into the next analysis |
| `run.py memory --rebuild` | Rebuild the ledgers from the archive |
| `run.py open <id>` | Decrypt and print a transcript |
| `run.py open <id> --kind audio --out f.mp3` | Recover the original recording |
| `run.py open <id> --kind analysis` | Decrypt and print the structured analysis |
| `run.py audit` | Read the compliance audit log |
| `run.py audit --recording-id <id>` | Everything that happened to one recording |
| `run.py audit --actor human --out audit.csv` | Export the human decisions |
| `run.py followups` | **Commitments still open, oldest first** |
| `run.py followups --draft open` | Draft the chase-ups into the outbox (never sends) |
| `run.py followups --done <id>` | Mark one done so it stops resurfacing |
| `run.py review` | What the review cadence says is due right now |
| `run.py review --reaffirm father` | Record a standing-consent reaffirmation |
| `run.py release <id>` | Release a quarantined recording after review |
| `run.py retention` | Dry-run the expiry sweep |
| `run.py retention --execute` | Actually delete expired artifacts |
| `run.py profiles` | Show the routing table |
| `run.py new-profile <id>` | Scaffold a new profile from the template |
| `run.py voices` | Show the installed voice packs |
| `run.py speakers list` | Who this archive can recognise by name |
| `run.py speakers enroll "Marcus" --audio clip.wav` | **Teach it a voice** |
| `run.py speakers identify <audio>` | Score a recording, change nothing |
| `run.py speakers forget "Marcus"` | Delete a voiceprint permanently |

`pip install -e .` installs the same commands as `plaud-bridge`, if you would
rather not type `python run.py` from the project directory.

### When it cannot actually hear anything

Speech recognition does not decline. Given music, a restaurant, a recital, or a
device that spent an hour in your pocket, Whisper returns fluent, confident,
well-punctuated English that nobody said. That is its best-documented failure
mode, and it is not a rough transcript with mistakes in it.

That matters here more than it would in a transcription tool, because nothing
downstream treats a transcript as provisional. It gets routed, promises get
extracted from it, memory carries those into next month's prompt, and the
worklist puts them in front of you as things you owe a client.

So the recogniser's own confidence is read. A transcript it was guessing at is
marked in the audit log and announced in the digest **above** the analysis —
everything below that line came out of it — and the extraction prompt is told to
prefer empty fields over interpretation.

Nothing is deleted and nothing is refused. A quiet conversation in a car scores
badly and is still the conversation you wanted.

**The thresholds under `asr.confidence` are guesses.** They were chosen from the
shape of the score distribution, not from your recordings. Run a few real files
through and read `open <id> --kind transcript` against the audio before trusting
either the warnings or their absence.

Imported text reports *unknown* rather than clean, because it carries no scores
and calling it clean would claim a check that never ran.

### From any recorder or app

The inbox takes what note-taking tools actually hand you, not just what one
brand exports:

| You have | Drop in the inbox |
|---|---|
| Plaud / Otter export | `.mp3` `.wav` `.txt` `.srt` |
| Apple Voice Memos, Google Recorder | `.m4a` |
| WhatsApp / Telegram voice note | `.opus` `.oga` `.amr` |
| Zoom / Teams / Meet **recording** | `.mp4` `.mov` `.webm` — the audio track is extracted |
| Zoom / Teams / Fireflies / YouTube **transcript** | `.vtt` |
| Typed or pasted notes | `.txt` `.md` |

Transcripts skip ASR entirely and everything downstream runs normally — the
cheap path when the exporting tool's transcription is already good enough, and
worth benchmarking before you commit to running your own.

One format is better than it looks: **Teams-style `.vtt` arrives with named
speakers.** Its `<v Name>` voice tags are the meeting platform stating who
spoke, from its own per-participant audio channels, and that attribution flows
through unchanged — no diarization, no enrollment, no model involved.

Anything with an extension nobody's recorder produces is refused by name rather
than silently ignored, and adding a format is one line in `pipeline.yaml`:
ffmpeg normalises any container, so the extension list is the only gate.

---

## The profiles

| Profile | Sensitivity | Cloud allowed | Consent required | Transcript kept | Audio kept |
|---|---|---|---|---|---|
| Insurance Agent | high | no | **yes** | 7 years | 1 year |
| Sales Trainer | medium | yes (redacted) | no | 5 years | 180 days |
| Father | maximum | **never** | family rule | 10 years | 30 days |
| Husband | maximum | **never** | spousal rule | 180 days | 14 days |
| Unfiled | high | no | no | 180 days | 30 days |

Father and Husband are locked to local processing in **code**, not just config.
Setting `allow_cloud_llm: true` in their YAML raises a configuration error at
startup with an explanation. Opening them up requires editing
`CODE_ENFORCED_LOCAL_ONLY` in `src/plaud_bridge/config.py`, which leaves a
commit behind. That friction is the feature.

### How the cloud actually gets used

Only Sales Trainer permits a cloud provider, so in the shipped configuration
almost nothing goes out:

- **Transcription is local unless you ask for cloud.** ASR runs before there is
  any text to classify, and a filename like `REC0042.wav` says nothing about
  what is on the recording. So cloud ASR is opt-in per file: name the file after
  a profile that allows it (`sales_trainer-roleplay.mp3`) and it goes to Groq.
  Everything else is transcribed locally. This is the expensive default on
  purpose — a slow local transcription is recoverable, an upload is not.
- **Routing is local.** The routing call sees the whole transcript before
  anything is known about it, so it stays local unless every profile permits
  cloud. Change that by permitting cloud on your profiles, not by editing code.
- **Analysis follows the governing profile**, not the profile being analysed. A
  recording that is Husband and Sales Trainer at once is analysed locally for
  both, because Husband governs the whole file.

The practical consequence: without a local LLM configured, the family profiles
fail rather than falling back. `run.py doctor` tells you this before you find
out the hard way.

### Editing a profile

Everything about a profile is config. Adding an extraction field is a YAML edit:

```yaml
extraction:
  fields:
    - key: "referral_asked"
      label: "Referral Asked"
      type: "boolean"
      description: "Did the producer ask for a referral before the meeting ended?"
```

Add it, rerun, done. No code change.

### Adding a profile

```bash
python run.py new-profile mentor --name "Mentor" --heading "Mentorship"
```

That copies `config/profiles/_TEMPLATE.yaml`, which documents every key inline,
with the ids filled in. Give it keywords and an `llm_hint` — without them the
router has nothing to go on and the profile will never match anything.

Files starting with an underscore are ignored by the loader, so the template
itself never becomes a profile.

---

## Named speakers

Diarization can tell three people apart. It has never heard any of them before,
so on its own the best it can write is `Speaker 2`. Enroll someone once and
their turns come back with their name on them.

```bash
# One clip of one person talking, thirty seconds is plenty
python run.py speakers enroll "Marcus" --audio clips/marcus.wav

# Trim to a clean stretch if the clip has other voices in it
python run.py speakers enroll "Dana" --audio meeting.m4a --start 42 --end 96

# A second clip from a different room makes matching more reliable
python run.py speakers enroll "Marcus" --audio clips/marcus-car.wav

python run.py speakers list
python run.py speakers forget "Marcus"
```

Before trusting it on real recordings, see what it actually thinks:

```bash
python run.py speakers identify data/inbox/2026-07-14.mp3
```

That prints every cluster, how long it spoke, and its similarity to everyone
enrolled. It writes nothing. It is the only sane way to pick a threshold for
your own voice, your own room, and your own microphone.

Enroll yourself too. `diarization.owner_label` guesses that the most-present
voice is the wearer, which is usually right and occasionally embarrassing; an
enrolled voiceprint beats the guess.

### It would rather say nothing than guess

Two guards stand between a similarity score and a name on a transcript:

- **Threshold.** The score has to clear `diarization.identify.threshold`
  outright. Nobody in the room being enrolled is the normal case, and the
  nearest of five strangers is still a stranger.
- **Margin.** The best match has to beat the runner-up by
  `diarization.identify.margin`. Relatives sound alike. Two brothers at 0.61 and
  0.60 means the model cannot tell them apart on this audio, which is a reason
  to stay quiet rather than flip a coin.

Anyone unmatched stays `Speaker 2`. A wrong name is worse than no name, because
a name is believed — it gets read six months later as fact, quoted into a
follow-up, and acted on. Silence about who spoke is recoverable.

A person is also used at most once per recording. The same voice cannot be two
people in the same room, so the higher score keeps the name.

### Where voiceprints live

In the vault, encrypted under your passphrase, as `voiceprints.enc`. There is no
plaintext fallback and no flag to ask for one: a voiceprint is biometric data
about someone who usually is not you, and a plaintext copy of it is a biometric
database sitting in a user directory. Enrollment without a working passphrase
refuses rather than degrading.

Nothing is uploaded. The embedding model runs locally, exactly like diarization.
For an air-gapped machine, fetch it with the rest:

```bash
python scripts/fetch_models.py --embedding      # or --all
```

`speakers forget` deletes the voiceprint. Transcripts already written keep the
name they were given, because rewriting history is not this tool's job — the
audit log and the archive are meant to be stable.

---

## Voice

The digest is the thing you actually read, so how it reads is config.

```bash
python run.py voices                    # what is installed, and what is active
python run.py digest --format html      # self-contained page, prints cleanly
```

Three packs ship in `config/voice/`:

| Pack | For |
|---|---|
| `plain` | Neutral and factual. The default. |
| `brief` | Short and scannable, for reading on a phone between appointments. |
| `warm` | Written like a person assembled it. Suits `--profile father` / `husband`. |

Switch with `voice.preset` in `pipeline.yaml`. Change a single line without
copying a whole pack:

```yaml
voice:
  preset: "plain"
  overrides:
    digest:
      needs_you:
        heading: "Before You Do Anything Else"
```

Profiles carry their own voice too — `digest.intro` opens a section,
`digest.empty` is what it says when the window turned up nothing, and
`extraction.persona` sets the register for that profile's analysis without
touching the hard constraints in its `system_prompt`.

**What voice cannot do.** It supplies words, never structure. Suppressed fields
still never render, personal profiles still stay out of the combined digest, and
a flagged recording still says so bluntly. A digest is the document most likely
to be forwarded, so the rules about what appears in it stay in code where a YAML
edit cannot reach them.

---

## Getting things back out

An archive you cannot search is a filing cabinet you cannot open.

```bash
python run.py search "own occupation" --content
python run.py search "biopsy" --content --profile husband --context 2
```

### Ask it a question

Search finds the phrase you remembered. `ask` answers the question you actually
had.

```bash
python run.py ask "what did I promise the Hendersons about their term policy?"
python run.py ask "what has Marcus been coached on?" --profile sales_trainer
python run.py ask "what did we agree about the school run?" --include-personal
python run.py ask "what is outstanding?" --save     # keep it, encrypted
```

Retrieval runs first and is deterministic — it ranks recordings by term overlap
and recency with no model involved. Only then is a model asked, and only about
the excerpts that were retrieved.

**Every citation is checked against what was actually sent.** If the model cites
a recording that was never in the bundle, that citation is dropped and the
answer says so by id. This is the failure mode that makes "ask your notes"
features untrustworthy, and it is the one thing here that is tested by deleting
the check and confirming the tests go red.

With no model configured at all, `ask` still works: you get the ranked excerpts
and a sentence saying plainly that this is search output rather than an answer.
It exits 2 when the answer is incomplete — a bounded scan, a trimmed context, a
dropped citation — and 0 when the honest answer is "nothing in the archive
matched", because that is a complete answer that happens to be nothing.

The strictest profile in the bundle decides whether the call may leave the
machine, exactly as ADR-002 decides it for a recording. Personal profiles stay
out unless you ask for them.

### Follow-ups, and drafts you send yourself

```bash
python run.py followups                      # what is still owed, oldest first
python run.py followups --profile insurance_agent
python run.py followups --draft open         # write drafts into the outbox
python run.py followups --done fu_3a91c2     # stop it resurfacing
```

The same promise made across three recordings collapses to one item, aged from
when you first made it. An eleven-day-old commitment to a client sorts above
yesterday's.

`--draft` writes a message into `data/outbox/drafts/`. **Nothing is sent. There
is no send path in the code at all** — no SMTP, no API, no mail configuration to
fill in. That is the deliberate half of the feature: the useful part is having
the message written, and the part worth refusing is a tool that mails a summary
of a private conversation on your behalf.

Drafts are redacted before they are written, even for a profile that has
`redact_before_llm` turned off. A draft is an outbound document by definition,
so a profile relaxing redaction for its own analysis does not relax it here.

### Learning across recordings

Every recording used to be analysed as if it were the first one ever seen.

```bash
python run.py memory                    # what it knows, per profile
python run.py memory --brief            # the briefing the next analysis will see
python run.py memory --rebuild          # throw it away and replay the archive
```

Each profile keeps a ledger of the people, open commitments, and recurring
topics it has heard, and the next analysis for that profile is made knowing
them. The ledger is **derived, never authoritative**: it is built from analyses
already stored, costs no model call, and `--rebuild` reproduces it from the
archive. If rebuild ever could not, the ledger would have quietly become a
second uncontrolled copy of your recordings.

Profile isolation is enforced by the cipher, not by convention. Each ledger is
encrypted under its own AAD, so what the Husband profile knows cannot decrypt
into an Insurance Agent prompt.

Entries decay, because a brief full of things that stopped being true crowds out
the ones that did not. A commitment closes only when a later recording says it
was done — deciding a promise was kept because its words came up again would be
inventing. And `forget` reaches memory too, or the command's promise would be
false.

`--content` searches what was said, decrypting the vault where it has to, and
prints the timestamp and speaker of every hit. It scans **everything** in the
window by default; `--scan-limit N` bounds the work and then says the answer is
incomplete.

**A search that could not look says so and exits non-zero** — whether a file
would not decrypt or a bound stopped it early. Concluding a phrase was never
said, when really nothing opened it, is the worst thing a search over your own
archive can do to you.

### Verify

```bash
python run.py verify
```

Opens every artifact the index points at. Missing files, silent corruption, and
a wrong passphrase all show up here. **An encrypted archive you have never tried
to decrypt is one you might already have lost** — this is the command that tells
you while it is still fixable. It also lists files on disk the index does not
know about, and never deletes anything.

If the vault is locked it reports encrypted artifacts as *unchecked*, not as
healthy. It will not claim a clean bill it could not confirm.

`verify` also counts the files that live in the vault and the outbox without
belonging to any recording — enrolled voiceprints, saved answers, drafts — and
names them as such rather than listing them as debris. Anything else in there is
still reported as an orphan, which is the point: a `verify` you have learned to
skim is worse than none.

### Export

```bash
python run.py export --days 30 --out handover.md
python run.py export --days 30 --transcripts --format html --out notes.html
```

The digest is written for you and assumes you are the only reader. An export is
the opposite: redaction applied, suppressed fields still never included,
personal profiles refused unless you pass `--include-personal`, and a footer
that states plainly that redaction is pattern matching rather than a guarantee.
That footer prints even when nothing matched, because "no pattern fired" is not
the same as "nothing sensitive is in here".

### Forget

```bash
python run.py forget rec_1a2b3c4d
```

Deletes one recording completely: vault artifacts, outbox files, the archived
original, the quarantine folder, and the index entry. It shows you the exact
file list and requires you to type `FORGET`.

The audit log keeps a record that the deletion happened, by a human, at a time.
Everything else goes. An audit trail that forgets deletions is not a trail.

---

## Back it up, and get it back

The vault has no key recovery — that is the point of a vault — which makes a
dead disk the one failure that loses everything. So back up:

```bash
python run.py backup                       # one encrypted .pbb file, safe anywhere
python run.py backup --out /mnt/drive/     # straight onto the external disk
python run.py restore plaud-backup-*.pbb   # refuses to overwrite without --force
```

The bundle holds the vault, the index, the outbox, the quarantine, and your
tuned config, encrypted whole with the vault's own cipher — safe to sit on a
drive or in a cloud folder this tool does not control. Restoring needs the
same passphrase; there is no backup of the backup's key, on purpose. Put the
passphrase in your password manager and put `backup` in your scheduler next to
`run`.

## When the gate holds a batch of recordings

Processing an old backlog means recordings from before you announced every
call, and the consent gate will hold them — correctly. Triage at scale:

```bash
python run.py quarantine                   # everything held, with the reason
python run.py quarantine --release-all     # bulk release; typed confirmation
python run.py quarantine --forget-all      # bulk delete, through real forget
```

`--release-all` never includes an explicit refusal. Someone who said "don't
record this" is not a batch decision; those release one at a time with
`release <id>`, or not at all.

## The clickable app

Prefer a window? `python desktop_app.py` opens the same pipeline in your
browser — passphrase, an Offline/free-cloud-key brain switch, drop recordings,
Process, Open digest. The Library doubles as a player: click a recording and
the original audio streams straight out of the vault (decrypted in flight,
never written to disk in the clear) with the transcript following the
playhead. Brief, People, and Insights tabs put the memo, the roster, and the
talk-time numbers a click away. `packaging/README.md` covers building the
double-click Windows `PlaudBridge.exe`. The HTML digest opens with charts
either way: minutes per section, activity across the window, and spend.

Before committing a big backlog to local processing, measure this machine:

```bash
python scripts/bench.py one-recording.mp3 --backlog-hours 200
```

---

## Running it without remembering to

```bash
python run.py watch --interval 300     # poll the inbox until you stop it
```

Or hand it to your scheduler. `run` is idempotent — content-hash dedupe means a
file already processed is skipped, so running it too often costs nothing:

```cron
*/15 * * * *  cd /path/to/plaud-bridge && .venv/bin/plaud-bridge run
0    7 * * 1  cd /path/to/plaud-bridge && .venv/bin/plaud-bridge digest --days 7 --format html --out data/outbox/week.html
0    9 1 * *  cd /path/to/plaud-bridge && .venv/bin/plaud-bridge review
```

The passphrase has to reach the process. A cron job with no
`PLAUD_BRIDGE_PASSPHRASE` will transcribe and then refuse to write anything
encrypted, which is the correct failure but a confusing one to debug at 7am.

---

## The review cadence

`COMPLIANCE.md` asks you to read certain things weekly, monthly, quarterly, and
annually. Asking a person to hold a four-tier schedule in their head is asking
them to stop by March, so:

```bash
python run.py review
```

assembles all four and tells you what is actually due: standing consent
reaffirmations that have lapsed, `statements_needing_review` across your client
calls, unfiled recordings with the keywords that would have routed them, and
artifacts past their expiry. It reports; it never deletes.

---

## Digest behaviour worth knowing

- **Personal profiles are omitted from the combined digest by default.** A
  combined digest is the document most likely to get forwarded, pasted, or
  opened on a shared screen. Pass `--include-personal` or `--profile father`
  when you actually want them.
- **Suppressed fields never render.** Client health and financial disclosures
  stay in the encrypted analysis file. The digest tells you how many exist. It
  does not print them.
- **"Needs You" comes first.** Anything flagged for human attention, plus every
  `next_action` across every section, before the detail.

---

## Costs

Transcription on Groq Whisper Turbo runs about `$0.04` per audio hour, so 20
hours a month is under a dollar — but see above: cloud ASR is opt-in per file,
so most recordings cost nothing to transcribe and take longer instead.

Analysis is priced from the token usage each provider reports, at the
`usd_per_million_*_tokens` rates in `pipeline.yaml`. That covers routing (one
call per recording) and extraction (one per matched profile).
`run.py status` shows cumulative spend, and `cost.warn_usd_per_run` /
`cost.halt_usd_per_run` stop a runaway loop.

**Verify current pricing yourself** rather than trusting the numbers baked into
the config. They were plausible when written; that is all anyone can say about
API pricing. A provider left unpriced counts as `$0.00` and says so in the log,
which means the halt threshold cannot see it.

---

## Read these before your first real recording

- `COMPLIANCE.md` — consent, state law, retention, and the things this tool
  cannot do for you
- `GOVERNANCE.md` — why the architecture is shaped this way, and what to check
  before changing it

---

## Tests

```bash
python -m pytest tests/ -q          # 523 tests
python scripts/smoke.py             # every CLI route, end to end
```

523 tests, no network and no API keys required.

`scripts/smoke.py` is the other half. It stands up a throwaway project in a temp
directory, drops transcript fixtures in its inbox, serves its own model on
loopback, and drives **every route a person can reach** as real subprocesses —
no ffmpeg, no model weights, no keys, and nothing written anywhere near your own
`data/`, which it checks byte for byte before and after.

The route list is read from the parser at runtime, so a subcommand added without
coverage fails the run as `route not covered` rather than quietly shrinking what
"every route" means. That has already caught two commands landing without
coverage, and two real defects that only appeared once a quarantined recording
was in the archive: a search that reported it as unreadable when it had simply
never been written, and `run --force` orphaning a second quarantine folder under
an id no index knew about.

The ones that matter most are in **`test_privacy_guarantees.py`**. Every test
there corresponds to a sentence this README states as a promise: a family
recording never reaching a cloud provider, the strictest profile governing the
whole file, the index holding no plaintext copy of an encrypted transcript, a
refusal never counting as consent. A failure in that file is a privacy
regression, not a bug.

After that: `test_config.py` for the local-only locks, `test_end_to_end.py` for
the spine, `test_cost_and_audit.py` for the spend ceiling and the audit trail,
`test_ingest_and_logging.py` for the quiet failures — the ones where nothing
raises and the transcript is simply missing words — and
`test_voice_and_templates.py`, which includes the check that no voice pack can
talk the digest into printing a suppressed field.

## License

This is proprietary software. Copyright (c) 2026 Hunger4Crypto
(hunger4crypto@gmail.com); all rights reserved. It is published for the owner's
own reference only — it is **not** open source, and no right to use, copy, run,
modify, or distribute it is granted to anyone else. See [LICENSE](LICENSE) for
the full terms. Being able to see this repository does not grant a license to
use it.
