"""Drishti: video-first automatic number plate recognition."""

__version__ = "0.1.0"

from .config import load_config  # noqa: E402

__all__ = ["load_config", "__version__"]


def __getattr__(name):  # lazy import keeps `import drishti` light
    if name == "Drishti":
        from .pipeline import Drishti

        return Drishti
    raise AttributeError(name)
