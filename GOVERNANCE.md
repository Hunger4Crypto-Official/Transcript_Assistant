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

## ADR-005: Routing runs before the compliance gate, conservatively

**Problem.** The gate needs to know which profiles matched to decide how
strictly to treat the file. But routing itself uses an LLM call, which is the
exact thing the gate is supposed to govern. Bootstrapping problem.

**Decision.** Before routing, run the free keyword prescore. If any
maximum-sensitivity profile shows signal above 0.15, the routing call itself
runs local-only. Filenames are checked the same way before ASR.

**Trade-off.** Some business calls get routed locally when a keyword coincides.
Slower, free, harmless. The reverse error is not harmless.

**Where.** `pipeline.py:_route` and `pipeline.py:_filename_suggests_sensitive`.

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

## Known limitations

1. **Crosstalk breaks diarization.** When two people talk over each other,
   speaker attribution degrades badly. No current system handles this cleanly.
   Budget for it to be the thing that surprises you.

2. **Plain-text transcript import has synthetic timestamps.** Imported `.txt`
   has no real timeline, so one is generated at an average speaking rate. Do not
   quote those timestamps as evidence of when something was said. `.srt` import
   carries real timestamps.

3. **Redaction is regex.** See `COMPLIANCE.md` section 4.

4. **Consent detection is phrase matching.** It catches common phrasings. Novel
   phrasing produces a false quarantine, which is the safe direction. Add
   patterns to `compliance/consent.py:_ANNOUNCE` if you find a gap.

5. **Cost estimates are for ASR only.** LLM token costs are not modelled per
   provider. `run.py status` shows what the providers reported, and the halt
   threshold is the real guardrail.

6. **This runs on one machine.** No multi-device sync, no server. That is a
   deliberate scope boundary, not an oversight.

---

## Before you change something

- [ ] Run `python -m pytest tests/ -q` first. Know the baseline is green.
- [ ] If it touches routing, compliance, or providers, run the end-to-end tests
      specifically. They encode guarantees that unit tests do not.
- [ ] If you added a provider, verify `is_cloud` is correct.
- [ ] If you relaxed a compliance control, write a new ADR here explaining why.
      Future you will want the reasoning, not just the diff.
