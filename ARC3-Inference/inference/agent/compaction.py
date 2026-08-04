"""LLM-based history compaction (C1): summarize dropped history blocks into a
rolling digest instead of silently truncating them.

Populated in Phase 2; this module must import cleanly from Phase 0 on.
"""

from __future__ import annotations
