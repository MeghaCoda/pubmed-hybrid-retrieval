from pathlib import Path


def find_project_root() -> Path:
    """Walk up from this file to the folder that contains pyproject.toml."""
    # __file__ is this file's own path; .parents lists every folder above it, nearest first.
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("No pyproject.toml found above paths.py")


DATA_DIR = find_project_root() / ".data"
EMBEDDINGS_CACHE = DATA_DIR / "embeddings.pkl"
QDRANT_PATH = DATA_DIR / "qdrant"
