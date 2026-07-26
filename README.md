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
| `run.py search "elimination period"` | Find recordings |
| `run.py open <id>` | Decrypt and print a transcript |
| `run.py open <id> --kind analysis` | Decrypt and print the structured analysis |
| `run.py release <id>` | Release a quarantined recording after review |
| `run.py retention` | Dry-run the expiry sweep |
| `run.py retention --execute` | Actually delete expired artifacts |
| `run.py profiles` | Show the routing table |

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
hours a month is under a dollar. Analysis cost depends on which model you point
at it. `run.py status` shows cumulative spend, and `cost.warn_usd_per_run` /
`cost.halt_usd_per_run` in `pipeline.yaml` stop a runaway loop.

Verify current pricing at groq.com/pricing rather than trusting the numbers
baked into the config. They were accurate when this was written; that is all
anyone can say about API pricing.

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

31 tests. The ones that matter most are in `test_config.py` (the local-only
locks) and `test_end_to_end.py` (a family recording never reaching a cloud
provider, missing consent producing a quarantine, personal content staying out
of the combined digest). If you change routing or compliance, run these first.
