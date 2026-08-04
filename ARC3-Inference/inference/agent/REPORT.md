# Context compaction (feat/context-compaction) — analysis & deploy report

Branch: `feat/context-compaction`, built off `main` (NOT off
`feat/meta-calibration`). All checks are reproducible:

```
uv sync --locked --extra dev
uv run pytest tests/compaction -q     # 19 passed
```

## 1. What already existed vs what this branch adds

| Mechanism | Status upstream | This branch |
|-----------|-----------------|-------------|
| Retained reasoning across steps | **Already present** — `assistant_message["reasoning"]` is stored into persistent history and resent verbatim in the next request payload | Verified by `tests/compaction/test_reasoning_retained.py` (end-to-end with a stubbed endpoint); no code change |
| Scientist notes (`_summarized_knowledge`) | Present — self-reported by the model via `_extract_scientist_note`; unreliable (only as good as the model's own report) | Untouched; keeps working |
| Rolling history window | Present — last 30 assistant turns (`_PERSISTENT_HISTORY_ASSISTANT_TURNS`) | Untouched |
| Context-budget trimming | Present — `_trim_messages_for_context` **silently drops** the oldest history block; same for `_force_reduce_messages` and the `context_overflow_recovered` path | **Upgraded**: every dropped block is first routed through an LLM compaction call into a rolling `HistoryDigest` (guaranteed layer underneath the self-reported layer) |
| Knowledge wipe on transition | Present — single condition wipes on `level_transition` / `run_complete` / `game_over` | Split behind `wipe_knowledge_on_level_transition` and `wipe_knowledge_on_reset` flags (defaults preserve upstream exactly; `run_complete` always wipes). The **digest** is never wiped on level transitions — only on a new game |

Design points (C1): one small compaction call per trim event (temperature
0.2, `max_tokens` 900, timeout 20 s, same endpoint/headers/client as the
analyzer); deterministic fallback to upstream silent-drop on any failure
(`compaction_fallback` logged/counted); drops smaller than
`min_dropped_tokens_to_compact` (800) skip the call entirely. The digest is
injected as a delimited `PERSISTED HISTORY DIGEST (auto-compacted)` block
adjacent to the scientist notes.

## 2. Measured truncation pressure (why this matters)

Dry-run over `example-run` transcripts (pass 0 of all 25 games, analysis
transcripts as per-turn history growth, 31,744-token budget, 30-turn
window):

- analyzer calls per run: mean ≈ 47
- **budget-overflow drop events per run: mean 43.4, median 43, max 72**
- runs with ≥1 silent truncation: **25/25**

i.e. upstream, once the window saturates (a few turns in), the agent loses
its oldest turn nearly **every step** — exactly the failure mode the public
harness reports describe (re-solving the game from scratch as insights fall
off the back of the context).

## 3. Expected overhead

With drops batched per trim invocation, compaction adds roughly one small
call per post-saturation step: ≈ 40 calls/run upper bound. At ~2–5 s per
900-token call this is ~1.5–3.5 min of a 45-min run (3–8 % wall-clock);
`min_dropped_tokens_to_compact` and the single-call-per-trim batching keep
it at the low end. The A/B (Section 5) measures the real cost — wall-clock
inside compaction is logged per run.

## 4. Instrumentation (C4)

Per analyzer step, the transcript carries a `COMPACTION` section with
cumulative `compaction_events`, `compaction_fallbacks`, `digest_tokens`,
`prompt_tokens` (estimate of the last request), `generated_tokens`, and
`compaction_wallclock_s`. Per run, a `compaction_summary:` line is emitted
on session close with totals. "Fewer output tokens, more coherent play" are
both measurable from these logs plus the standard score files.

## 5. Live A/B instructions

1. Baseline: `compaction.enabled=false` (as shipped), full game set,
   n_passes ≥ 20 → `<baseline>/score.json`.
2. Candidate: identical config with `"compaction": {"enabled": true}` →
   `<candidate>/score.json`. Same seeds/config otherwise.
3. Compare with the existing tooling (do not reimplement):

   ```
   uv run inference-significance \
       --baseline <baseline_run> --candidate <candidate_run> \
       --bootstrap-samples 10000
   ```

4. Metrics: score (paired per-game bootstrap + permutation), actions/run,
   generated tokens/run, plus `compaction_events`/`fallbacks`/wall-clock
   from the C4 logs.
5. We make no claim about the externally reported uplift figures; those come
   from a different model, API, and harness. Our claim is only what this A/B
   measures.

## 6. CRITICAL — interaction with feat/meta-calibration

> **meta-calibration must be recalibrated on post-compaction trajectories
> before both features are enabled together.**

This branch changes the agent's trajectory distribution (longer coherent
memory ⇒ different first-clear hazard, different no-op/novelty profiles).
Any restart-policy parameters calibrated on pre-compaction trajectories
(T thresholds, mixture parameters π/p_E/p_H, hazard coefficients) are stale
under compaction. Recommended sequencing:

1. merge compaction;
2. regenerate trajectories (a full multi-pass run with compaction enabled);
3. re-run the meta-calibration pipeline on the new data
   (`extract → survival → mixture → simulate`);
4. only then enable both features together.

The one deliberate coupling point implemented here for the restart branch:
the world-model wipe on reset/level-transition is now configurable
(`wipe_knowledge_on_reset`, `wipe_knowledge_on_level_transition`), with
defaults equal to upstream behavior.

## 7. Acceptance status

- [x] Phase 1 — retained reasoning verified end-to-end (no new code needed).
- [x] Phase 2 — compaction unit tests green, including LLM-failure fallback.
- [x] Phase 3 — parity golden: with `enabled=false` the full request-payload
      stream is byte-identical to the pre-wiring agent (SHA-256 fingerprint).
- [x] Phase 4 — smoke test: ≥2 compactions fire; an early marker fact
      survives truncation into the injected digest; digest stays ≤ 700
      tokens; every posted request stays within the context budget.
- [ ] Live A/B (Section 5) — the only remaining gate before enabling by
      default.
