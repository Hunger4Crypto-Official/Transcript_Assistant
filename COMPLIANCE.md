# COMPLIANCE.md

Operational policy for this repository. Written down so it survives you
forgetting it in six months.

**This is not legal advice and this tool is not a lawyer.** Verify anything
here with counsel before relying on it. What follows is how the software
behaves and the reasoning behind it.

---

## 1. Consent

### The rule that makes everything else moot

**Announce every time. Get a verbal yes on tape. Every call, every state.**

It costs eight seconds and it removes the entire risk category. Do not build a
workflow that depends on knowing which state the person on the other end is
sitting in, because you often will not.

A script that works:

> "Before we get started, I record these conversations so I can focus on you
> instead of taking notes. Is that okay with you?"

Wait for the answer. Get it on the recording. Then proceed.

### What the software checks

The consent detector scans the first 90 seconds for two things:

1. You announcing the recording
2. **A different speaker** agreeing

Both are required. You agreeing with yourself does not count, and there is a
test that enforces that. If both are not found on a profile that requires
consent, the recording is **quarantined**: not analysed, not indexed, copied to
`data/quarantine/<id>/` with a `WHY.md` explaining what to do.

To release one after you have listened and confirmed consent was actually
obtained:

```bash
python run.py release <recording_id>
```

The release is written to the audit log with `actor=human`.

### What the software cannot check

It tells you whether you **said the words**. It cannot tell you whether the
consent was legally sufficient where the other party was sitting. Statutes
change, get reinterpreted, and differ for in-person versus telephone.

### State posture

| State | General posture | What you do |
|---|---|---|
| Nevada | Telephone communications generally require all parties (NRS 200.620) | Announce, get a yes |
| Florida | All-party | Announce, get a yes |
| Texas | One-party | Announce, get a yes |
| Arizona | One-party | Announce, get a yes |

The right column is identical on purpose.

Configured all-party list lives in `compliance.all_party_consent_states`.
**Verify current statutes with counsel.** Nevada's telephone rule in particular
has a messy interpretive history.

---

## 2. Family and spousal recording

### Father profile

The `family_consent.everyone_knows_device_records` flag must be `true`. If you
set it to `false`, the profile refuses to process and the file is quarantined.

That flag is a house rule, not a statute. The device is visible and the family
knows it records. **If that stops being true, stop using this profile.** A
config change is not the correct response to that situation.

### Husband profile

`spousal_consent.she_knows_and_agreed` must be `true`. Read this part carefully:

In an all-party consent state, **your spouse is a party**. Marriage is not an
exception to the statute. Recording her without her knowledge is a legal
exposure and, separately, a bad idea.

### What the extraction prompts forbid

Both family profiles carry hard constraints in their system prompts. The model
is instructed **not** to:

- Assess, diagnose, score, or psychologically profile a child
- Assign blame or judge parenting quality
- Evaluate or score the relationship
- Take a side in a disagreement
- Summarise an argument at all

If a family recording is primarily conflict, or contains anything suggesting a
child is at risk, the model returns `requires_human_attention: true` and
nothing else. The pipeline discards whatever else came back and the digest says
"read it yourself."

That is deliberate. **A conversation between people who love each other is not
an artifact to be indexed and re-litigated later.** The tool captures
commitments, logistics, and things worth remembering. Nothing more.

---

## 3. Processing locality

| Profile | ASR | LLM | Enforced by |
|---|---|---|---|
| Father | local only | local only | **code** (`CODE_ENFORCED_LOCAL_ONLY`) |
| Husband | local only | local only | **code** |
| Insurance Agent | local only | local only | config |
| Unfiled | local only | local only | config |
| Sales Trainer | cloud ok | cloud ok | config, redacted first |

Insurance Agent is local-only by default because client conversations routinely
surface health and financial disclosures, and no BAA is in play at a standard
developer API tier. If you obtain one and want to relax this, change it in
`config/profiles/insurance_agent.yaml` and write down why in `GOVERNANCE.md`.

When compliance requires local processing, cloud providers are **removed from
the provider chain**, not deprioritised. If nothing local is available the run
fails loudly. It does not fall back. A tool that quietly degrades its own
security guarantee is worse than one that stops.

---

## 4. Redaction

Applied to the copy of the transcript handed to a model, never to the stored
transcript. The stored transcript is your record; redacting it would destroy the
thing you are trying to keep.

Patterns configured: SSN, credit card, policy number, date of birth, email,
phone, and spoken digit strings ("five five five one two three...").

Regex redaction is a floor, not a ceiling. It catches a spoken SSN in standard
form and misses one spoken unusually. Treat it as defence in depth behind the
real control, which is not sending sensitive profiles to a cloud provider at all.

---

## 5. Retention

| Artifact | Insurance | Trainer | Father | Husband |
|---|---|---|---|---|
| Transcript | 7 years | 5 years | 10 years | 180 days |
| Raw audio | 1 year | 180 days | 30 days | 14 days |
| Audit log | 7 years | 2 years | 1 year | 1 year |

**Audio expires far sooner than transcripts everywhere.** Audio is the
liability; transcripts are the asset. Ten years of transcripts costs you
megabytes. Ten years of audio costs you a discovery request.

The Husband profile is deliberately short. A multi-year indexed archive of your
marriage is an asset with no upside and real downside. If you disagree, change
it, and write down why.

Sweeps are dry-run by default:

```bash
python run.py retention              # show what would go
python run.py retention --execute    # actually delete, with confirmation
```

Confirm the insurance retention period against your current E&O carrier and
whatever upline or BGA relationship you are operating under now. The 7-year
default is a common expectation, not a universal requirement.

---

## 6. Discoverability

Every recording is a potential exhibit.

If a client later disputes what was represented in a life or disability sale,
that file exists. This cuts both ways: it protects you when you did everything
right, and it buries you when you got sloppy on a single sentence. The
`statements_needing_review` field exists for exactly this reason. It flags
anything that could read as a guarantee, a projection, a tax or legal opinion,
a carrier comparison claim, or a securities recommendation.

Read that field. Every week. It is the highest-value output in the system.

The same logic applies to family recordings, in a different register. An
indexed, analysed archive of household conversations is discoverable in a
custody or divorce context. That is the primary reason the Husband retention
window is 180 days rather than years.

---

## 7. The vault

Everything at high or maximum sensitivity is encrypted at rest with AES-256-GCM,
key derived by scrypt from `PLAUD_BRIDGE_PASSPHRASE`. The key is never written
to disk.

**There is no recovery.** Lose the passphrase and the vault is gone. That is
correct behaviour for a vault. Put it in your password manager before you
process anything you would be upset to lose.

If encryption is unavailable, the pipeline **refuses to write** rather than
falling back to plaintext. You will notice a crash. You would not notice a quiet
plaintext write.

---

## 8. Audit log

Every ingest, route, compliance decision, quarantine, release, and retention
deletion is written to the `audit` table with a UTC timestamp. Releases from
quarantine are marked `actor=human`.

Transcript content is never written to application logs. Logs carry
identifiers, hashes, counts, and durations. You can hand a log file to someone
for debugging without handing over anything private.

---

## 9. Review cadence

- **Weekly** — read `statements_needing_review` in the Production section
- **Monthly** — read the Unfiled section and add the keywords the router missed
- **Quarterly** — dry-run the retention sweep, confirm the plan, execute
- **Annually** — reconfirm consent posture with counsel; reconfirm the family
  and spousal consent flags are still honest
