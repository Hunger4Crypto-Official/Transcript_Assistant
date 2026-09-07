"""
The smoke suite, in CI.

`scripts/smoke.py` is the only thing in this repository that proves the shipped
command line works the way a person reaches it: real subprocesses, real config
on disk, real encryption, no test harness holding it up. That makes it exactly
the kind of script that rots, because a script nobody runs is documentation
with a shebang on it. So it runs here, on every commit, and a route that breaks
turns this file red.

The wrapper uses the `--quick` subset: every route still runs, but only the one
canonical check per route rather than every flag combination. That keeps the
pytest run near ten seconds while preserving the claim that matters -- that
every route a person can reach still works end to end. The full flag matrix is
what `python scripts/smoke.py` is for.

The other half of what is asserted here is the honesty property. The route list
comes from `build_parser()` at runtime, so a subcommand added to the CLI without
a matching entry in the smoke coverage table fails this file rather than
silently going untested.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from typing import Any

import pytest

from _fixtures import ROOT
from plaud_bridge.cli import build_parser


def smoke():
    """The suite is a script, not a package. Import it the way a user runs it."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import smoke as module

    return module


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict[str, Any]:
    """
    One full `--quick` pass, shared by every assertion below.

    Module scope on purpose: the run stands up a project, processes four
    recordings and drives twenty-odd subprocesses, and doing that once per
    assertion would turn a ten second test into a minute of the same work.
    Standard output is captured rather than left to capsys so the fixture is
    not tied to function scope.
    """
    root = tmp_path_factory.mktemp("smoke-project")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = smoke().main(["--quick", "--json", "--tmp-dir", str(root)])
    payload = json.loads(buffer.getvalue())
    payload["exit_code"] = code
    return payload


def _failures(report: dict[str, Any]) -> list[str]:
    return [
        f"{route['route']}: {check['command']} -> {check['detail'] or check['status']}"
        for route in report["routes"]
        for check in route["checks"]
        if check["status"] == "FAIL"
    ]


def test_every_route_passes_or_says_why_it_cannot(report):
    assert not _failures(report), "\n".join(_failures(report))
    for route in report["routes"]:
        assert route["status"] in ("OK", "SKIPPED"), route


def test_the_suite_itself_exits_zero(report):
    """The exit code is what CI reads, so it is asserted separately."""
    assert report["exit_code"] == 0
    assert report["ok"]


def test_no_route_in_the_parser_is_left_uncovered(report):
    assert report["uncovered"] == [], (
        "these subcommands exist but scripts/smoke.py does not exercise them: "
        f"{report['uncovered']}. Add an entry to ROUTES there in the same commit "
        "that adds the route."
    )


def test_it_leaves_the_real_data_directory_alone(report):
    """
    The repository's own data/ is somebody's actual archive.

    A suite that writes into it while claiming to be safe on a laptop has done
    the one thing it promised not to.
    """
    assert report["data_dir_drift"] == [], report["data_dir_drift"]


def test_the_analysis_really_ran_against_the_local_stub(report):
    """
    Routing and extraction happened, over a socket, with no network.

    Zero completions would mean the recordings were processed without any model
    call at all, which is a run that proved much less than it appears to.
    """
    assert report["stub_completions"] > 0


def test_the_table_and_the_parser_agree_in_both_directions():
    """
    The completeness claim, available in a millisecond and without a sandbox.

    A route in the parser and not in the table is untested; a route in the table
    and not in the parser is coverage of something that no longer exists. Both
    are wrong and the message says which happened.
    """
    module = smoke()
    declared, covered = set(module.parser_routes()), set(module.ROUTES)
    assert declared == covered, (
        f"not covered by scripts/smoke.py: {sorted(declared - covered)}; "
        f"covered there but gone from the CLI: {sorted(covered - declared)}"
    )


def test_every_skipped_check_states_a_reason():
    """
    A skip is a claim that something cannot run headlessly, and a claim with no
    reason attached is how a route quietly stops being tested.
    """
    module = smoke()
    for name, checks in module.ROUTES.items():
        for check in checks:
            if not check.skip:
                continue
            assert len(check.skip.split()) >= 4, f"{name}: {check.label} skips without saying why"


def test_the_routes_it_covers_are_the_ones_the_parser_declares():
    """
    Guards the discovery itself.

    Everything above trusts `parser_routes()`, so if that ever stopped seeing
    the subcommand table it would report perfect coverage of nothing.
    """
    module = smoke()
    declared = set(build_parser()._subparsers._group_actions[0].choices)
    found = module.parser_routes()
    # Nested groups appear as "parent child", so compare on the first word.
    assert {name.split()[0] for name in found} == declared
    assert len(found) >= len(declared)
