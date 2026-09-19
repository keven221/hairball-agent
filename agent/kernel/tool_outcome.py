"""One decision about whether a tool result is a failure.

Both kernel paths must answer this the same way. They previously did not: the v1
tap used a prefix heuristic while the v2 engine parsed JSON and looked for an
``error`` key, so the same tool output could be a failure on one path and a
success on the other. That makes the two paths incomparable in an A/B run and,
worse, understates failures on whichever path is blinder — and the ok flag feeds
doom-loop detection and tool settlement, so under-reporting weakens loop control
rather than just the metrics.

The predicate is the union of both earlier checks, so neither path loses a
failure it used to catch:

* structured — a JSON object carrying a truthy ``error``, or ``success: false``
  (a shell command that exits non-zero reports the latter and no ``error`` key)
* textual — output that *starts* with an error marker

It stays prefix-anchored on the textual side on purpose. A successful result is
allowed to mention the word "error" in its body (a log excerpt, a diff, a test
report), and scanning the whole text would fail those.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_TEXT_MARKERS = ('{"error"', '{"success": false', "error:", "❌")


def looks_like_error(output: Any) -> bool:
    """Whether *output* reads as a failed tool result."""
    if not output:
        return False
    if isinstance(output, Mapping):
        parsed = output
        text = ""
    else:
        text = output if isinstance(output, str) else str(output)
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None
    if isinstance(parsed, Mapping):
        if parsed.get("error"):
            return True
        # ``success`` is only meaningful when the tool actually reports it;
        # a missing key must not read as failure.
        if "success" in parsed and not parsed.get("success"):
            return True
        return False

    head = text.lstrip()[:120].lower()
    return any(head.startswith(marker) for marker in _TEXT_MARKERS)
