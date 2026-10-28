import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pkg(alias: str, rel_path: str):
    """Import a service's `app` package under a unique name (every service calls it `app`)."""
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(
        alias, path / "__init__.py", submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def sub(alias: str, name: str):
    return importlib.import_module(f"{alias}.{name}")
