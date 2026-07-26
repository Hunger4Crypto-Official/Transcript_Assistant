# Plaud Bridge

Takes audio exported from a Plaud recorder and turns it into transcripts,
per-profile analysis, and a digest you can read or filter. Runs on your machine.
Nothing about a family conversation ever leaves it.

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
| `run.py status` | Index summary: counts, hours, spend |
| `run.py watch` | Process the inbox on an interval until you stop it |
| `run.py search "elimination period"` | Find recordings by filename |
| `run.py search "own occupation" --content` | **Search what was actually said** |
| `run.py verify` | Confirm every artifact still exists and still decrypts |
| `run.py export --days 30` | Build a redacted document for someone else |
| `run.py forget <id>` | Permanently delete one recording |
| `run.py open <id>` | Decrypt and print a transcript |
| `run.py open <id> --kind analysis` | Decrypt and print the structured analysis |
| `run.py audit` | Read the compliance audit log |
| `run.py audit --recording-id <id>` | Everything that happened to one recording |
| `run.py audit --actor human --out audit.csv` | Export the human decisions |
| `run.py review` | What the review cadence says is due right now |
| `run.py review --reaffirm father` | Record a standing-consent reaffirmation |
| `run.py release <id>` | Release a quarantined recording after review |
| `run.py retention` | Dry-run the expiry sweep |
| `run.py retention --execute` | Actually delete expired artifacts |
| `run.py profiles` | Show the routing table |
| `run.py new-profile <id>` | Scaffold a new profile from the template |
| `run.py voices` | Show the installed voice packs |

`pip install -e .` installs the same commands as `plaud-bridge`, if you would
rather not type `python run.py` from the project directory.

You can also drop a Plaud-exported **transcript** (`.txt` or `.srt`) into the
inbox instead of audio. ASR is skipped entirely and everything downstream runs
normally. That is the cheap path if their transcription is already good enough
for you, and it is worth benchmarking before you commit to running your own ASR.

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

`--content` searches what was said, decrypting the vault where it has to, and
prints the timestamp and speaker of every hit. **If a recording cannot be
decrypted it says so and exits non-zero** rather than returning fewer results —
concluding a phrase was never said, when really the file would not open, is the
worst thing a search over your own archive can do to you.

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
python -m pytest tests/ -q
```

128 tests, no network and no API keys required.

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
