"""
A furnished archive, so the first five minutes are not an empty room.

Every read-only view this tool has -- the digest, the brief, the roster, the
talk-time numbers, the follow-up worklist, the quarantine triage -- is a view
over recordings. On a fresh install there are none, so a new person meets
seven empty tabs and has to record their life for a week before they can tell
whether any of it is useful. That is a bad trade to ask of someone deciding
whether to trust a tool with their conversations.

This writes a small set of sample transcripts into the inbox, which the normal
pipeline then processes exactly like anything else -- no special path, no
pre-baked database, no fixture the real code would never see. What you look at
afterwards is the real machinery working on transcripts that happen to be
fictional.

Three rules hold everything here honest:

  - **Fiction is labelled as fiction.** Every sample's filename starts with
    `sample-`, its first line says it is a sample, and the people in it have
    names no real archive would collide with. Nobody should discover a month
    later that "Dana Whitfield" was never a client.
  - **Nothing real is touched.** Files are only ever written into the inbox,
    never over an existing file, and a name already taken is skipped rather
    than clobbered. `--clean` removes only files this module wrote, matched by
    that same `sample-` prefix and verified against the catalogue below.
  - **The owner speaks as the owner.** The samples substitute the configured
    `diarization.owner_label`, so the roster marks you as you and the talk-time
    numbers are about the right person. A demo that shows a stranger's speaking
    habits teaches nothing.

The set is chosen to populate every state a reader might want to see: an
encrypted client call with consent given (the ordinary case), a coaching
session that stays plaintext at rest (a different retention posture), a family
conversation that is forced local and excluded from shared digests, and one
call with no consent exchange at all -- which the gate quarantines, because
somebody evaluating this tool should see the gate work rather than read about
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .logging_setup import get

log = get("demo")

# The token replaced with the configured owner label in every sample.
_OWNER = "{OWNER}"

# The banner every sample carries. It is a transcript line, so it survives
# into the stored transcript and shows up in `open`, search results, and the
# player -- there is nowhere a reader can meet this content without it.
_BANNER = "[SAMPLE] This is fictional sample data created by `run.py demo`, not a real recording."


@dataclass(frozen=True)
class Sample:
    """One fictional recording, and what it exists to demonstrate."""

    filename: str
    shows: str
    body: str


CLIENT_CALL = Sample(
    "sample-client-fact-find.txt",
    "the ordinary case: consent given, routed to work, encrypted at rest",
    f"""\
{_BANNER}
{_OWNER}: Morning Dana, before we get into it -- I record these calls for my own notes. Is that all right with you?
Dana Whitfield: Yes, that's fine.
{_OWNER}: Appreciate it. Tell me what you have in place today.
Dana Whitfield: Just a small term policy through work. Two hundred thousand, I think.
{_OWNER}: And is there any disability coverage, through the employer or private?
Dana Whitfield: No, nothing like that. That's really why I called.
{_OWNER}: That is the gap I would look at first. Your income is the asset, not the house.
Dana Whitfield: I had not thought about it that way. What drives the cost on something like that?
{_OWNER}: The elimination period mostly, and whether you want an own occupation definition.
Dana Whitfield: The premium is my worry, honestly. Things are tight.
{_OWNER}: That is fair, and I would rather right-size it than oversell you. Let me build two options.
Dana Whitfield: Send them over and I will look this weekend.
{_OWNER}: I will have them to you by Thursday. Can you email me your date of birth before then?
Dana Whitfield: Yes, tonight.
""",
)

COACHING_SESSION = Sample(
    "sample-coaching-roleplay.txt",
    "a different posture: no consent requirement, plaintext at rest, so the player can scrub it",
    f"""\
{_BANNER}
{_OWNER}: Let's run a role play on objection handling, then review your pipeline activity.
Priya Raman: Ready. Give me the hardest objection you get.
{_OWNER}: Here it is: the premium is too high, I need to think about it.
Priya Raman: I would slow down and ask what specifically feels high before I talk about price.
{_OWNER}: Good instinct. That is the rapport framework doing the work rather than a script.
Priya Raman: My activity was ninety dials this week and four appointments set.
{_OWNER}: The dials are fine. The close rate is where I would spend your next month.
Priya Raman: Can you watch a recording of my next discovery call and mark it up?
{_OWNER}: Send it over Monday and I will go through it line by line.
""",
)

FAMILY_EVENING = Sample(
    "sample-family-evening.txt",
    "the locked case: forced local, encrypted, and left out of shared digests",
    f"""\
{_BANNER}
{_OWNER}: How was practice today?
Sam: Coach said I am starting on Saturday.
{_OWNER}: That is great. What time do you need to be there?
Sam: Nine, and you have to sign the permission slip for the field trip.
{_OWNER}: I will sign it tonight. Remind me at bedtime.
Sam: Can we get pizza after the game like you said?
{_OWNER}: I did say that. Pizza after the game.
""",
)

NO_CONSENT_CALL = Sample(
    "sample-no-consent-call.txt",
    "the gate working: a client call with no consent exchange, held for review",
    f"""\
{_BANNER}
{_OWNER}: So walk me through the coverage you have in place right now.
Marcus Bell: A term policy through work, about two hundred thousand.
{_OWNER}: And a disability policy anywhere?
Marcus Bell: No. The mortgage is the thing that worries me, about four hundred thousand left.
{_OWNER}: Then the elimination period is where I would start, and an own occupation definition.
Marcus Bell: Send me something in writing and I will read it this weekend.
""",
)

SAMPLES: tuple[Sample, ...] = (
    CLIENT_CALL, COACHING_SESSION, FAMILY_EVENING, NO_CONSENT_CALL,
)

# Filenames this module is allowed to delete. `--clean` checks membership here
# rather than globbing `sample-*`, so a real recording a person happened to
# name "sample-something" is never removed by a cleanup they did not aim at it.
SAMPLE_NAMES = frozenset(s.filename for s in SAMPLES)


def owner_label(cfg) -> str:
    """The configured owner label, or a neutral stand-in if it is unset."""
    return str(cfg.get("diarization.owner_label", "") or "").strip() or "You"


def render(sample: Sample, owner: str) -> str:
    """One sample's text with the owner label substituted in."""
    return sample.body.replace(_OWNER, owner)


def install(cfg, *, overwrite: bool = False) -> tuple[list[Path], list[Path]]:
    """
    Write the samples into the inbox. Returns (written, skipped).

    Skipping rather than overwriting is the safe default: the inbox is a place
    a person drops real recordings, and a demo command is not a reason to
    replace one. `overwrite=True` exists for re-running the demo deliberately.
    """
    inbox = cfg.path("inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    owner = owner_label(cfg)
    written: list[Path] = []
    skipped: list[Path] = []
    for sample in SAMPLES:
        dest = inbox / sample.filename
        if dest.exists() and not overwrite:
            skipped.append(dest)
            continue
        dest.write_text(render(sample, owner), encoding="utf-8")
        written.append(dest)
        log.info("demo: wrote %s", dest.name)
    return written, skipped


def clean(cfg) -> list[Path]:
    """
    Remove sample files still sitting in the inbox.

    Only unprocessed samples: once the pipeline has taken one, removing the
    inbox copy would do nothing useful anyway, and what a person actually wants
    then is `forget <id>`, which this deliberately does not do on their behalf.
    """
    inbox = cfg.path("inbox")
    removed: list[Path] = []
    for name in sorted(SAMPLE_NAMES):
        path = inbox / name
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def describe() -> str:
    """What the set contains, for the command's own output."""
    lines = ["The samples, and what each one is there to show:", ""]
    for sample in SAMPLES:
        lines.append(f"  {sample.filename}")
        lines.append(f"      {sample.shows}")
    return "\n".join(lines)
