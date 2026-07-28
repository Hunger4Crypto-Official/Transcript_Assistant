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

## ADR-017: One decryption path

**Decision.** Everything that reads stored content back — content search,
verification, export, deletion — goes through `archive.py`. The digest is the
one exception and it is on the list to fold in.

**Why.** A recording's words live in one of two places depending on its
governing profile: in the SQLite payload when it is not encrypted, and only in
the vault when it is (ADR-013). Every reader has to know that rule. Four copies
of it is four chances for one to forget the encrypted case and silently return
nothing, which reads to the user as "that was never said".

**The rule for readers.** Under-reporting is worse than failing. `search_content`
returns matches *and* a list of what it could not open, and the command exits
non-zero when that list is non-empty. `verify` marks a locked vault's artifacts
as `unchecked` rather than omitting them, because reporting "0 artifacts
indexed" when there are twelve is how someone concludes an archive is fine when
nothing was looked at.

---

## ADR-018: Deletion is a first-class operation, and the audit survives it

**Decision.** `forget <id>` removes the vault artifacts, outbox files, archived
original, quarantine folder, and index row for one recording. The audit entries
for that recording stay.

**Why deletion at all.** Retention sweeps on a schedule; that is not the same as
being able to remove one specific conversation because someone asked you to, or
because it should never have been recorded. Without this the only way to delete
a single recording was hand-editing SQLite, which nobody does correctly at 11pm.

**Why the audit stays.** The `audit` table deliberately has no foreign key onto
`recordings`, so rows survive the delete. "This recording was deleted, by a
human, at this time" is exactly what an audit trail is for. A trail that forgets
deletions is not a trail. The entry is written *before* the files are removed,
so a crash halfway through still leaves evidence.

---

## ADR-019: Unsafe defaults are the ones that ship

**Decision.** `ComplianceVerdict.encrypt_at_rest` defaults to `True`. A
recording is treated as needing encryption until a profile says otherwise.

**Why.** The verdict is only filled in once the gate has run, and a recording
can fail before that — a provider outage, a malformed profile, a bug. With the
old `False` default, that path wrote the complete plaintext transcript of a
family conversation into `bridge.db` and left it there permanently. Nothing
raised. The run reported `failed=1` and looked like a normal bad day.

**The general rule.** When a flag governs whether content may sit in the clear,
the default is the safe one, and the unsafe value is set explicitly by the code
that has actually established it is safe. Any default is a decision about what
happens on the paths nobody thought about.

**Related.** `is_encrypted` now reads that single field. It used to be derived
from sensitivity while persistence used the profile's `encrypt_at_rest` flag —
two sources of truth for one question, which a perfectly legal profile (high
sensitivity, encryption off) made disagree.

---

## ADR-020: Destructive operations are bounded to our own directories

**Decision.** `forget` and the retention sweep resolve every candidate path and
refuse anything outside the configured `vault`, `outbox`, `inbox`, `quarantine`,
and `work` directories. Refusals are logged, audited, and reported.

**Why.** Deletion targets come from the index, and the index is a file. It can
be restored from a backup taken when paths meant something else, hand-edited, or
corrupted. Before this, `retention --execute` would unlink whatever it was told
to. Verified: a row pointing at an unrelated PDF outside the data directory
deleted the PDF.

**Cost.** A legitimately relocated data directory now needs the index updated
rather than silently following the old paths. That is the right way round.

---

## ADR-021: A search that did not look must not report a result

**Decision.** `search_content` returns what it scanned, what it skipped, and
what would not open. The command exits non-zero when the answer is incomplete,
and scans everything by default.

**Why.** The CLI's `--limit` reads as "how many results to show" and was being
handed to the row query as "how many recordings to open". An archive of 60
recordings had 10 searched and 50 silently excluded, and the command printed
*nothing matching "elimination period" was said*. The phrase was in the archive.

A tool that answers "that never happened" when it means "I did not look" is
worse than one with no search at all, because you believe it.

---

## ADR-022: A speaker is named only when the model is not close to wrong

**Decision.** Identification puts a name on a diarized cluster only when the
similarity clears an absolute threshold AND beats the runner-up by a margin. A
person is used at most once per recording. Everything else stays `Speaker N`.

**Why.** A name is believed. `Speaker 2` is read as a placeholder and checked
against memory; "Marcus" is read as fact, quoted into a follow-up, and acted on
six months later by someone who was not in the room. The two failure modes are
not symmetric, so the guards are not symmetric either.

The margin exists because the nearest-neighbour framing is misleading on the
recordings this tool is actually for. Family members sound alike. So do a father
and a son on a phone speaker. When two enrolled people score 0.61 and 0.60, the
model has not identified anybody; it has produced a tie and a rounding error.

The one-name-per-recording rule follows from the same reasoning: a single voice
cannot be two people in one room, so if two clusters both want the same name,
at most one of them can be right.

---

## ADR-023: Voiceprints are encrypted or they are not stored

**Decision.** Enrollment requires a working vault passphrase. There is no
plaintext fallback, no `--force`, and no config key to ask for one. Without a
passphrase, `speakers enroll` refuses.

**Why.** Everywhere else in this project, an unavailable vault degrades to
"process the low-sensitivity profiles only". That reasoning does not transfer.
A voiceprint is biometric data about people who did not install this software
and mostly do not know it exists — clients, kids, a spouse. A plaintext
`voiceprints.json` is a biometric database in a user directory, backed up to
wherever that directory syncs.

The consistent choice with ADR-019 is that the unsafe option is the one that
does not ship at all.

---

## ADR-024: A citation names something that was actually sent, or it is dropped

**Decision.** `ask` validates every citation the model returns against the
bundle it was given. A citation naming a recording that was not in the bundle is
dropped and reported by id. A citation whose timestamp does not exist is snapped
to the retrieved excerpt whose words best match the quote, never to a number the
model chose.

**Why.** The whole value of answering from an archive is that the answer is
anchored to something that was said. A fabricated citation inverts that: it
makes an invented claim *more* believable than an uncited one, because it comes
with a recording id and a timestamp that look checkable and are not.

This is the one failure mode that would make the feature worse than the search
it replaces, so it is verified by deleting the check and confirming the tests go
red rather than by reading the code and being satisfied.

---

## ADR-025: Drafting has no send path, by construction

**Decision.** `followups --draft` writes a file into the outbox. There is no
SMTP client, no mail API, no address book, and no configuration key that would
enable one. Drafts are redacted before they are written regardless of the
profile's `redact_before_llm` setting.

**Why.** The feature this replaces auto-summarises a meeting and mails it out.
The useful half is having the message written; the half worth refusing is
software deciding, unattended, that a summary of a private conversation should
leave the machine and go to a named person. A confirmation prompt is not the
answer, because the failure is not "the user did not notice", it is "the user
noticed on the fourth of four occasions".

Unconditional redaction diverges from `export`, deliberately. A profile turning
redaction off is a statement about its own analysis, which stays here. A draft
is outbound by definition.

---

## ADR-026: Memory is derived, never authoritative

**Decision.** The per-profile ledgers are built only from analyses already
stored, hold no content the archive does not, and can be discarded and rebuilt
from the archive at any time. `memory --rebuild` reproducing the ledger is a
test, not a convenience. `forget` clears memory as part of the same command.

**Why.** A ledger that could not be rebuilt would have become a second copy of
your recordings — one that no retention sweep expires, `verify` never checks,
and `forget` does not reach. That is the exact shape of the thing this project
exists to avoid, arrived at by accident rather than by decision.

Profile isolation is enforced by encrypting each ledger under its own AAD rather
than by the code being careful. Care is a property of the code as written today;
a decryption that fails is a property of the file.

---

## ADR-027: A commitment closes only when something says it was done

**Decision.** An open commitment is closed by a later recording that names it as
completed, in a declared closure field. It is never closed by its words coming
up again, by time passing, or by a similar commitment appearing.

**Why.** Both errors are possible and only one is recoverable. A commitment left
open after it was kept is visible: it sits in `followups` and you close it. A
commitment closed because the topic was mentioned again disappears silently, and
what disappears is a promise you made to a client or a child.

---

## ADR-028: A transcript the recogniser was guessing at says so first

**Decision.** Every segment's average log probability and no-speech probability
are read after transcription. A transcript scoring badly is marked, audited, and
announced in the digest **above** the analysis, and the extraction prompt is
instructed to prefer empty fields over interpretation. Nothing is deleted and no
recording is refused.

**Why.** Speech recognition does not decline. Given music, a restaurant, a
recital, or a device in a pocket, it returns fluent, well-punctuated English
that nobody said. That is not a rough transcript with mistakes in it; it is
invented text, identical in shape to the real thing.

That matters more here than it would in a transcription tool, because nothing
downstream treats the transcript as provisional. The router files it, the
extractor pulls promises out of it, memory carries those promises into next
month's prompt, and the worklist puts them in front of you as things you owe a
client. A hallucinated sentence does not stay a sentence. It becomes a
commitment you believe you made, six months after the audio is gone.

The scores have been collected since the first version and read by nothing.

**Constraints.** Judgement is weighted by duration, not by segment count: four
minutes of invented music is one segment and twenty honest interjections are
twenty, and counting them equally lets the thing that matters lose the vote. A
transcript with no scores at all -- imported text -- is reported as *unknown*
rather than clean, because calling it clean claims a check that never ran. The
warning is placed last in the prompt, beside the instruction, since a caveat
given as background gets noted and then extracted from confidently anyway.

**What this is not.** It is not a quality gate. A quiet conversation in a car
scores badly and is still the conversation you wanted, and deciding a recording
is worthless is not a call to make automatically. The thresholds are guesses
that have been tuned against nothing; they are config, and they are worth
checking against your own microphone before either number is trusted.

---

## ADR-029: A quote is findable in the transcript, or it is dropped

**Decision.** Every field the schema types as a `quote` is checked against the
exact text the model was shown. Anything not present verbatim -- after
normalising case, punctuation, and whitespace -- is dropped and counted, not
flagged and kept.

**Why.** This is ADR-024 applied one layer up, and it matters more here. A
fabricated citation in `ask` sits next to an answer you are already reading
critically. A fabricated quote is attributed to a named person, flows into the
memory ledger as something they said, and can surface in a digest a year later
when the audio is gone. Nothing downstream re-checks it.

Checking against the text the model was **shown** — not the raw transcript — is
the load-bearing detail. Compliance redacts before the model sees anything, so
the model can only quote redacted text; validating against the original would
condemn every legitimate quote from a redacted recording.

**On dropping rather than flagging.** The schema calls the field a quote and the
prompt demands the speaker's exact words, so a passage that is not present is
not a quote — it is a paraphrase wearing quotation marks and a timestamp. An
empty field reads as "nothing worth keeping was said." An invented one reads as
testimony. Only one of those is recoverable.

---

## ADR-030: The prompt is built so the cache can work

**Decision.** A profile's system prompt, persona, and schema are sent as one
cached block; the transcript travels in the user turn and never inside it. No
sampling parameter is sent to Anthropic at all.

**Why.** Caching is a prefix match: the stable half has to come first, and one
changed byte ahead of the marker invalidates everything after it. The system
half is byte-identical across every recording and every episode of every
recording, so without this the same few thousand tokens are paid at full price
forever. Cache reads bill at a fraction of fresh input.

The sampling parameter is a separate story with the same shape. `temperature`
was pinned to 0.0 for determinism it never actually provided; on the current
models it is rejected outright, so it was not a harmless leftover but a 400 on
the first call after any model upgrade. It is gone, and the output contract in
the extraction prompt does that job instead.

**Constraint.** This is Anthropic-specific and deliberately not generalised. The
Groq path keeps its `temperature` and sends no `cache_control` — the parameter
removal happened on one vendor's models, not on every endpoint that speaks the
same wire format, and quietly "fixing" the other provider would change its
behaviour for no reason.

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

8. **Speaker identification degrades with the room, not gracefully.** A
   voiceprint enrolled from a quiet clip matches poorly against a car, a
   restaurant, or a phone speaker. Enroll two or three clips from the places you
   actually record. The default threshold is a starting point, not a
   calibration: run `speakers identify` on your own audio and read the scores
   before trusting any of it.

9. **Quote verification is exact, not fuzzy.** ADR-029 forgives case,
   punctuation, and whitespace and nothing else. A model that lightly rewords
   ("I will" for "I'll") has its quote dropped. That is the intended reading of
   a field typed `quote`, but it means the count is a measure of paraphrasing as
   well as of invention.

10. **The confidence thresholds are unvalidated.** ADR-028 reads the
   recogniser's own scores, but `-1.0` and `0.6` are starting points chosen
   from the shape of the distribution, not from your recordings. Run a few real
   files through and compare `open <id> --kind transcript` against the audio
   before trusting either the warnings or their absence.

11. **Crosstalk defeats identification the same way it defeats diarization.** A
   cluster containing two overlapping voices embeds to something that is neither
   of them, which the margin guard will usually reject. Usually.

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
