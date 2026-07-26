# GOVERNANCE.md

Architectural decisions and the reasoning behind them. Read this before you
change something that looks unnecessary. Most of what looks unnecessary here is
load-bearing.

---

## ADR-001: Multi-label routing, not single-label

**Decision.** A recording can match several profiles simultaneously.

**Why.** Real life is not categorised. A dinner conversation where a client
calls is genuinely Husband and Insurance Agent. Single-label routing forces a
loss: either the client follow-up disappears into a family folder, or a private
conversation gets filed under work.

**Cost.** It creates the collision problem that ADR-002 exists to solve.

---

## ADR-002: The strictest matched profile governs the entire recording

**Decision.** When several profiles match, the highest-sensitivity one sets the
processing policy for the whole file. Not per-segment. Whole file.

**Why.** The alternative is software deciding, mid-file, that the first half of
a conversation is safe to send to a third party and the second half is not.
That is not a decision software should make, and there is no reliable boundary
to draw.

**Consequence.** One private sentence in a business meeting locks the whole
recording to local processing and encryption. This is intended. If you find it
annoying, the fix is to record separately, not to relax the rule.

**How it is enforced.** The gate computes `governing_profile` and
`force_local_processing`, and `pipeline._analyse` reads them once and applies
them to every analysis on the recording. Per-profile policy is a floor, never a
ceiling: `extract()` takes the caller's answer and can only make it stricter.

This is worth stating explicitly because the original code did the opposite.
Each analysis recomputed locality from its own profile, so a recording matching
both Husband and Sales Trainer kept the Husband analysis local and then sent the
identical marital transcript to a cloud provider for the Sales Trainer one. The
gate's verdict was computed correctly and read by nothing. If you add a stage
that talks to a model, read the verdict.

**Config.** `compliance.strictest_profile_governs`. Setting it false is
supported and inadvisable.

---

## ADR-003: Family profiles are locked in code, not config

**Decision.** `father` and `husband` appear in `CODE_ENFORCED_LOCAL_ONLY` in
`src/plaud_bridge/config.py`. Setting `allow_cloud_llm: true` in their YAML
raises a `ConfigError` at startup with an explanation.

**Why.** Config gets edited at midnight while debugging something unrelated. A
guarantee that a config edit can silently remove is not a guarantee. Opening
these profiles up requires a source change, which leaves a commit behind and
gives you a chance to reconsider.

**How to actually change it.** Edit `CODE_ENFORCED_LOCAL_ONLY`, then write the
reason here as a new ADR, then commit both together.

---

## ADR-004: Local-only means providers are removed, not deprioritised

**Decision.** When the compliance gate requires local processing, cloud
providers are removed from the chain entirely. If nothing local is available,
the run fails.

**Why.** A fallback chain that ends at a cloud provider defeats the exclusion.
The failure mode of "your family recording did not process because
faster-whisper is not installed" is recoverable. The failure mode of "your
family recording was uploaded to a third party because the local provider was
missing" is not.

**Where.** `asr/registry.py:build_asr_chain`, `llm/registry.py:build_llm_chain`.
Both take `local_only` and filter on `provider.is_cloud`.

**If you add a provider**, set `is_cloud` correctly. Getting it wrong silently
defeats this entire control. There is no test that can catch it for you.

---

## ADR-005: Everything before the gate defaults to local

**Problem.** The gate needs to know which profiles matched to decide how
strictly to treat the file. But routing itself uses an LLM call, and ASR runs
even earlier, before there is any text to route on. Both are exactly the thing
the gate is supposed to govern. Bootstrapping problem.

**Decision.** Pre-gate stages do not try to guess the content. They apply the
policy every profile shares.

- **Routing** runs local-only unless *every* routable profile permits a cloud
  LLM. In the shipped config only Sales Trainer does, so routing is local.
- **ASR** is local unless the filename positively names a profile that permits
  cloud ASR *and* no locked profile is implicated. Cloud is opt-in per file.

**Trade-off.** Business calls get transcribed and routed locally unless you name
the file for it. Slower, free, harmless. The reverse error is not harmless.

**What this replaced, and why it is worth reading.** The original rule was the
inverse: cloud by default, local only when a keyword from a locked profile
happened to appear. A Plaud export is named `REC0042.wav`, which contains no
keywords at all, so a recording of your child went to a third-party
transcription service. The ADR text above this line was already correct about
the principle; the code implemented its mirror image. That gap survived because
the end-to-end test covering it fed a `.txt` file, which skips ASR entirely, and
named it `dinner-with-kid.txt`, which tripped the heuristic by luck.

**Where.** `pipeline.py:_route`, `pipeline.py:_asr_local_only`,
`config.py:cloud_llm_permitted_by_every_profile`. Pinned by
`tests/test_privacy_guarantees.py`.

---

## ADR-006: Deterministic glossary correction, not LLM correction

**Decision.** Post-ASR correction is a compiled regex lookup table in
`config/glossary.yaml`, not a model pass.

**Why.** An LLM correction pass is more flexible and will occasionally rewrite
something a client actually said. In a records context that is unacceptable.
Every change the glossary makes is auditable, reversible, and logged with a
count.

**How to grow it.** Every time you catch a mangled term, add it. It compounds.
The `asr_bias_terms` list is separately injected into the ASR decoder prompt to
prevent the error rather than repair it.

---

## ADR-007: Refuse rather than degrade

**Decision.** If `cryptography` is unavailable or the passphrase is unset, the
vault raises rather than writing plaintext.

**Why.** You will notice a crash. You will not notice a quiet plaintext write
until it matters, and by then it has been happening for months.

**Same principle elsewhere.** No ASR available under a local-only constraint:
fail. Consent required but not detected: quarantine. Cost ceiling exceeded:
halt. The system prefers stopping to guessing.

---

## ADR-008: stdlib-only core

**Decision.** No `requests`, `httpx`, `pydantic`, or ORM. `urllib`, `sqlite3`,
`dataclasses`, and PyYAML. Heavy dependencies (`faster-whisper`, `pyannote`,
`cryptography`) are isolated behind capability checks.

**Why.** This is a personal tool that should still install in five years on a
machine you have not touched. Every dependency is a future breakage. The
providers are optional and gracefully report their own absence.

**Cost.** More code in `http_util.py` than a one-line `requests` call. Worth it.

---

## ADR-009: Chunking with overlap, deterministic stitching

**Decision.** Audio is split into windows sized to stay under the ~25MB cloud
upload cap, with 8 seconds of overlap. The stitcher removes duplicated speech
from the overlap using normalised string similarity at 0.72.

**Why overlap.** A word cut in half at a boundary is lost in both chunks. With
overlap it survives in one of them.

**Why deterministic.** Same reason as ADR-006. The threshold is tuned to catch
ASR variation of the same sentence ("own occupation" vs "own-occupation.")
without collapsing genuine repetition later in the meeting. Both cases have
tests.

---

## ADR-010: Personal profiles excluded from the combined digest by default

**Decision.** `exclude_from_combined_export: true` on `father` and `husband`.
`--include-personal` or `--profile <id>` overrides it.

**Why.** The combined digest is the document most likely to be forwarded,
pasted into a message, or opened on a shared screen. The default assumes that
will happen.

---

## ADR-011: Suppressed fields never render to markdown

**Decision.** `suppress_fields` on the Insurance Agent profile keeps health and
financial disclosures out of every rendered digest. They remain in the
encrypted analysis JSON.

**Why.** You need to know a client disclosed a health condition. You do not
need it printed in a document you might read on a plane. The digest reports the
count and points at `run.py open <id> --kind analysis`.

---

## ADR-012: Extraction schemas live in config

**Decision.** Every profile's fields, types, descriptions, and system prompt
come from YAML. `_coerce` in `extractor.py` forces whatever the model returns
into the declared shape.

**Why.** Adding a field should be a two-minute edit, not a code change and a
test run. The coercion layer means a model returning a string where the schema
declared a list produces a one-item list rather than a crash.

**Constraint.** The hard constraints in the family profiles' system prompts are
not decoration. Read `COMPLIANCE.md` section 2 before editing them.

---

## ADR-013: The index is a plain file, so it holds no plaintext

**Decision.** For a recording whose governing profile encrypts at rest, the
SQLite index stores metadata only. The transcript segments and the extracted
analysis fields are withheld and marked as withheld. The vault artifact holds
them. `DigestBuilder` decrypts on demand to render them.

**Why.** `data/bridge.db` is an ordinary file with ordinary permissions that
gets copied around with the rest of `data/`. Writing the verbatim transcript
into it left an unencrypted copy of a maximum-sensitivity conversation sitting
next to the encrypted one, which makes the vault decorative. Retention made it
worse: the sweep unlinked the encrypted artifact and never touched the index, so
after the 180-day husband window the encrypted copy was gone and the plaintext
copy remained indefinitely.

**Cost.** Rendering a digest for an encrypted profile now needs the passphrase.
That is the correct trade and it is consistent with ADR-007: when the digest
cannot decrypt, it says so in the section rather than rendering an empty one.

---

## ADR-014: Spend is counted wherever it is incurred

**Decision.** LLM providers price their own calls from the token usage they
report, using rates in `pipeline.yaml`. Routing cost is carried out of the
router rather than discarded. Every exit path in `process_file` adds the
recording's cost to the run total, including quarantine and failure.

**Why.** `cost.halt_usd_per_run` is advertised as the thing that stops a runaway
loop. It could only see ASR spend, because no LLM provider ever set `cost_usd`
and the router dropped its response on the floor. A run that quarantined every
file still paid for a routing call per recording and reported `$0.00`.

**Constraint.** A provider with no configured rate contributes zero and logs
that it is doing so, once. An invented number inside a spend guardrail is worse
than a visible zero, because the zero is at least honest about not knowing.

---

## ADR-015: Voice is config; structure is code

**Decision.** Every user-facing string in the digest comes from a voice pack in
`config/voice/`. What renders, in what order, and under what conditions stays in
`digest/builder.py`. There is no template language and packs cannot introduce
control flow.

**Why not a real template engine.** The digest is the document most likely to be
forwarded, pasted into a message, or opened on a shared screen, and its renderer
is where three compliance rules are enforced: suppressed fields never print,
personal profiles are omitted from the combined view, and encrypted analyses are
decrypted on demand rather than mirrored into the index. A template able to
reorder or re-emit sections is a template able to defeat all three. Trading that
for layout flexibility is a bad trade in this specific document.

**Why it cannot fail.** `voice.py` carries the complete default set, and a pack
is a deep merge over it. A partial pack is valid, a missing pack falls back with
a warning, a corrupt pack falls back with a warning, and an unknown placeholder
renders empty. A typo in a voice file should make the digest look slightly
wrong, never lose you the digest.

**Where.** `voice.py`, `config/voice/*.yaml`, and per-profile `digest.intro`,
`digest.empty`, `extraction.persona`. Pinned by
`tests/test_voice_and_templates.py`, including a test that an override cannot
print a suppressed field.

---

## ADR-016: Prompt layering puts the constraints last

**Decision.** An extraction prompt is assembled in three layers: the voice
pack's `analysis.house_style`, then the profile's `extraction.persona`, then the
profile's `extraction.system_prompt`.

**Why that order.** The hard constraints on the family profiles — no
psychological assessment of a child, no fault-finding in a marital
disagreement — live in `system_prompt`. Putting them last places them nearest
the task and means nothing configured above can dilute them. House style and
persona set register; they are not allowed to argue with a rule.

**If you add a layer**, add it above, not below.

---

## Known limitations

1. **Crosstalk breaks diarization.** When two people talk over each other,
   speaker attribution degrades badly. No current system handles this cleanly.
   Budget for it to be the thing that surprises you.

2. **Plain-text transcript import has synthetic timestamps.** Imported `.txt`
   has no real timeline, so one is generated at an average speaking rate. The
   timeline is monotonic and explicit `[MM:SS]` markers are honoured, but do not
   quote those timestamps as evidence of when something was said. `.srt` import
   carries real timestamps.

3. **Redaction is regex.** See `COMPLIANCE.md` section 4.

4. **Consent detection is phrase matching.** It catches common phrasings, treats
   an objection anywhere in the window as decisive, and requires the
   announcement to come from `diarization.owner_label` when that speaker can be
   identified at all. Novel phrasing produces a false quarantine, which is the
   safe direction. Add patterns to `compliance/consent.py` if you find a gap —
   `_ANNOUNCE`, `_AGREE`, and `_REFUSE` are all worth growing.

5. **LLM cost is modelled from configured rates, not billed amounts.** The rates
   in `pipeline.yaml` are a guardrail, not an invoice, and they go stale. Verify
   them against current published pricing before relying on the halt threshold.
   Anthropic cache reads and writes are counted as plain input tokens, which is
   close but not exact.

6. **Speaker labels in imported text are inferred.** A `Name:` prefix is treated
   as a speaker when it repeats or reads like a name. A one-off label that also
   looks like a name will still be taken as a speaker.

7. **This runs on one machine.** No multi-device sync, no server. That is a
   deliberate scope boundary, not an oversight.

---

## Before you change something

- [ ] Run `python -m pytest tests/ -q` first. Know the baseline is green.
- [ ] If it touches routing, compliance, or providers, run
      `tests/test_privacy_guarantees.py` specifically. Every test in it
      corresponds to a sentence the README states as a promise. A failure there
      is a privacy regression, not a bug.
- [ ] If you added a provider, verify `is_cloud` is correct.
- [ ] If you added a stage that talks to a model, read
      `rec.compliance.force_local_processing`. Do not recompute locality from a
      profile you happen to be holding.
- [ ] If you relaxed a compliance control, write a new ADR here explaining why.
      Future you will want the reasoning, not just the diff.

A note on the ADRs above, several of which were rewritten after the code was
audited against them. The documents were right and the code did not implement
them, in four separate places, each of which read as reasonable in isolation.
The lesson worth keeping: an architectural guarantee that no test asserts is a
comment. If you write an ADR, write the test that fails when it stops being
true.
