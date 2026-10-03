import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _tiny_vcm_root() -> Path | None:
    p = Path(os.environ.get("TINY_VCM_ROOT", REPO_ROOT.parent / "tiny-vcm")).resolve()
    return p if (p / "src" / "vcm_data_loader.py").exists() else None


@pytest.fixture
def tiny_vcm_root() -> Path:
    root = _tiny_vcm_root()
    if root is None:
        pytest.skip("tiny-vcm checkout not found (set TINY_VCM_ROOT)")
    return root
