"""Single source for where the pipeline's data lives.

Every script resolves its inputs and outputs against PIPELINE (the `pipeline/`
folder), so `--data-dir data/<corpus>` means the same thing wherever a script
sits in the package and whatever directory it is run from.
"""
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
