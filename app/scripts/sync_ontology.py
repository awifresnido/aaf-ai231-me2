"""Copy tiny-vcm/configs/ontology.json into vcm_common/ (run after the training ontology changes)."""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "tiny-vcm" / "configs" / "ontology.json"
shutil.copy2(src, REPO / "vcm_common" / "ontology.json")
print(f"copied {src} -> vcm_common/ontology.json")
