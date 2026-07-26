import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from _fixtures import build_sandbox  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """The shipped config, redirected at a temp dir, with a stubbed LLM."""
    return build_sandbox(tmp_path, monkeypatch)
