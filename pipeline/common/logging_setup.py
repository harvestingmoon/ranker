"""One place that configures logging for every entry point.

Modules only do `logger = logging.getLogger(__name__)` and call `info`/`debug`
themselves; nothing else decides where the output goes. That keeps the library
importable without side effects (a plain `import pipeline.features.features`
does not touch the root logger) while every `main()` gets the same format.

`configure()` is idempotent: calling it twice does not double the handlers, which
matters because entry points call it and helpers might too.
"""
import logging
import sys
from typing import Optional

from pipeline.common import settings

_configured = False


def configure(level: Optional[str] = None) -> logging.Logger:
    """Install the pipeline's handlers on the root logger and return it.

    level overrides PIPELINE_LOG_LEVEL (so a --verbose flag can raise the
    verbosity of one run without editing .env).
    """
    global _configured
    root = logging.getLogger()
    if _configured:
        if level:
            root.setLevel(level.upper())
        return root

    root.setLevel((level or settings.LOG_LEVEL).upper())

    formatter = logging.Formatter(settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT)

    # Only add a stderr handler if nothing else (pytest, an embedding host) has
    # already claimed the root logger; otherwise we would duplicate every line.
    if not root.handlers:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if settings.LOG_FILE:
        from pathlib import Path

        Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
        to_file.setFormatter(formatter)
        root.addHandler(to_file)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Shorthand so modules do not each import logging plus configure()."""
    configure()
    return logging.getLogger(name)
