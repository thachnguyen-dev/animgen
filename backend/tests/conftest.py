import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import synth  # noqa: E402
from animgen.engines.demo import DemoEngine  # noqa: E402
from animgen.session import Session  # noqa: E402


@pytest.fixture
def hero(tmp_path) -> Path:
    return synth.character(tmp_path / "hero.png")


@pytest.fixture
def session(tmp_path) -> Session:
    s = Session(DemoEngine(), tmp_path / "data")
    yield s
    s.close()


@pytest.fixture
def sword(tmp_path):
    return synth.sword(tmp_path / "sword.png")
