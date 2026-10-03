"""Console encoding safety for the command-line tools.

On Windows (and anywhere output is redirected under a legacy locale) ``print()`` of Arabic text
raises UnicodeEncodeError. Every CLI calls ``ensure_utf8_output()`` first so Arabic prints
correctly whether the output goes to a terminal, a pipe, or a file.
"""

from __future__ import annotations

import sys


def ensure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass  # not a reconfigurable text stream (e.g. a test capture object): leave it alone
