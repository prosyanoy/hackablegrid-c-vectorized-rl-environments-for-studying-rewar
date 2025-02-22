import sys
from pathlib import Path

# Use the in-tree build (`make build`) when present; otherwise the pip-installed package.
_pkg = Path(__file__).resolve().parents[1] / "python"
if any((_pkg / "hackablegrid").glob("_core*.so")):
    sys.path.insert(0, str(_pkg))
