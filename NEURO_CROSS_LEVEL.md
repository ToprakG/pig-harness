# Göktürk V2: cross_level_notes reminder

Grounding: successor representation / complementary learning systems --
mechanics (transition structure) should survive a goal/layout change, only
the episodic/layout-specific part should reset.

IMPORTANT OVERLAP: `feat/level-debrief` already implements a related,
almost certainly more thorough mechanism (13 hits for
cross_level_notes/debrief in that branch vs this prompt-only reminder).
Check that branch FIRST. This commit is here for comparison / because it
was independently derived, not because it should be merged over
level-debrief. Do not run both -- pick one.

Change: `_summarized_knowledge_lines` now appends an explicit reminder once
`_levels_seen >= 1` and `cross_level_notes` is still empty, nudging the model
to write control-mapping/mechanics under that field instead of re-deriving
them from scratch on later levels. Also increments `_levels_seen` on each
`level_transition`.

Not run end-to-end.
