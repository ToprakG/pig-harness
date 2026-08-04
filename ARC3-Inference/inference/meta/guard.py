"""Genericity guard: whitelist policy inputs, forbid board/identity leakage.

Two checks, both hard failures (exit 1):

1. **Feature whitelist** — every declared policy-input feature name must be
   in ``WHITELIST``. Anything else (board content, game identity, colors,
   shapes, ...) is a leak: the entire value of this branch is that the
   policy conditions on generic run dynamics only.
2. **Source scan** — no module in ``inference/meta/`` may reference
   ``board_ascii`` / ``color`` / ``shape`` / ``game_id`` as policy inputs.
   ``extract.py`` is exempt by design: it is the single ingestion boundary
   whose job is to see ``board_ascii`` once and reduce it to a short hash.
   A line may opt out with a ``# guard-ok: <reason>`` pragma (use sparingly).

Usage::

    uv run python -m inference.meta.guard            # exit 0 clean, 1 dirty
"""

from __future__ import annotations

import argparse
import os
import re
import sys

WHITELIST = frozenset({
    "action_num",
    "level",
    "level_progress",
    "t_since_level",
    "noop_rate_w",
    "novelty_rate_w",
    "budget_frac",
    "reward_seen",
    "attempt_index",
    "posterior_tractable",
})

FORBIDDEN_TOKENS = ("board_ascii", "color", "shape", "game_id")
# extract.py: the ingestion boundary that reduces raw boards to hashes.
# guard.py: must spell the forbidden tokens to ban them.
EXEMPT_FILES = ("extract.py", "guard.py")
PRAGMA = "# guard-ok:"

_TOKEN_RES = {tok: re.compile(rf"\b{tok}\w*", re.IGNORECASE)
              for tok in FORBIDDEN_TOKENS}


def check_feature_names(names) -> list[str]:
    """Return violation messages for feature names outside the whitelist."""
    return [f"feature '{n}' is not in the policy-input whitelist"
            for n in names if n not in WHITELIST]


def scan_sources(root: str, exempt=EXEMPT_FILES) -> list[str]:
    """Scan ``*.py`` under ``root`` for forbidden tokens; return violations."""
    violations = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".py") or fn in exempt:
                continue
            path = os.path.join(dirpath, fn)
            with open(path) as f:
                for lineno, line in enumerate(f, 1):
                    if PRAGMA in line:
                        continue
                    for tok, rx in _TOKEN_RES.items():
                        if rx.search(line):
                            violations.append(
                                f"{path}:{lineno}: forbidden token '{tok}' "
                                f"in: {line.strip()}")
    return violations


def run_guard(root: str, feature_names) -> list[str]:
    return check_feature_names(feature_names) + scan_sources(root)


def main(argv: list[str] | None = None) -> int:
    from inference.meta.policy import POLICY_INPUT_FEATURES

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.dirname(__file__),
                    help="directory tree to source-scan")
    args = ap.parse_args(argv)

    violations = run_guard(args.root, POLICY_INPUT_FEATURES)
    if violations:
        for v in violations:
            print(f"GUARD FAIL: {v}", file=sys.stderr)
        return 1
    print("GUARD: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
