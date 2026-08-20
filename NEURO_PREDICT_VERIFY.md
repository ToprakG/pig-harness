# Göktürk V1: predict-then-verify

Grounding: predictive coding (learning driven by prediction error, not raw
input). `object_diff`-style signals were already computed elsewhere in this
codebase but nothing upstream of them issued an expectation, so the model
could not distinguish "world did what I expected" from "world surprised me" --
exactly the signal that should trigger belief revision.

Checked for overlap before writing this: no `CONFIRMED`/`VIOLATED`/predict
pattern found on `main`, `feat/meta-calibration`, `feat/stall-reflect`,
`feat/action-efficiency` (grep, 2026-08-20). Appears net-new.

Change: `PYTHON_ADDENDUM` in `prompts.py` now asks the model to state expected
effect before `action(...)` and print CONFIRMED/VIOLATED after. Zero new LLM
calls, zero new runtime plumbing -- prompt-only.

Not yet run end-to-end (no local GPU/arcengine). Needs a real pass before
trusting the effect size.
