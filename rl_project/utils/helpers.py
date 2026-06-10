import os
import random
import yaml
import numpy as np


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(device_str: str = "auto"):
    """Select compute device. 'auto' picks CUDA if available, else CPU."""
    import torch
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_config(path: str) -> dict:
    """Load a YAML config file and return as a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_env(filename: str = ".env"):
    """Load key=value pairs from a .env file into os.environ (no overwrite).

    Searches the current working directory and parent directories until a
    file named ``filename`` is found. Lines starting with '#' and blank lines
    are ignored. Values already present in os.environ are not overwritten.
    Returns the path that was loaded, or None if no .env was found.
    """
    start = os.path.abspath(os.getcwd())
    for directory in [start, *_parents(start)]:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            with open(candidate, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
            return candidate
    return None


def _parents(path: str):
    """Yield successive parent directories of ``path``."""
    while True:
        parent = os.path.dirname(path)
        if parent == path:
            return
        yield parent
        path = parent


def get_api_key(name: str = "OPENAI_API_KEY") -> str:
    """Return an API key from the environment, loading .env first if needed.

    Raises a clear error if the key is missing so scripts fail fast instead of
    sending an invalid key to the API.
    """
    if name not in os.environ:
        load_env()
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(
            f"{name} is not set. Add it to a .env file at the repo root "
            f"(e.g. {name}=sk-...) or export it in your shell."
        )
    return key
