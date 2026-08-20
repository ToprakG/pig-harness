# Neuroscience → ARC-AGI-3 agent design

Purpose: map what is known about biological learning onto the specific,
measured failures of our agent. Every section ends with a concrete change and
how we would measure it. Analogies that cannot be turned into a measurable
change are marked as such and dropped.

## 0. Why the biology is relevant here at all

ARC-AGI-3 is explicitly a test of *fluid* intelligence: acquiring a new skill
inside an unfamiliar, interactive environment within a small action budget.
The scoring formula makes this precise — `level_score = (human_actions /
agent_actions)^2` — so the benchmark is not asking "can you eventually solve
it" but "can you infer the rules from very few interactions."

That is the same computational problem the brain faces in a novel environment,
and it is the problem for which biology has evolved specialised machinery. The
useful part of the analogy is not "neural networks resemble neurons"; it is
that certain *algorithmic* solutions (fast episodic binding, replay,
predictive maps, gated working memory) are convergent answers to exactly this
sample-efficiency problem.

## 1. Complementary Learning Systems — and why context management *is* learning

Classic CLS theory (McClelland, McNaughton, O'Reilly): the hippocampus learns
fast, sparsely, one-shot; the neocortex learns slowly, in a distributed,
interleaved way. Fast learning without a slow system catastrophically
interferes; slow learning alone cannot acquire anything in one trial.

Our situation: the LLM's weights are frozen at submission time. There is no
slow-system update available. **Therefore the entire learning capacity of our
agent lives in its context window.** Everything the agent can ever learn about
a game is whatever survives in the prompt.

This reframes context management from an engineering nicety to *the* learning
mechanism. It also predicts, correctly, the largest published effect in this
space: OpenAI reported ARC-AGI-3 public-set score moving 13.3% → 38.3% purely
by retaining reasoning across steps and compacting rather than truncating
context. Discarding reasoning after each action is, in CLS terms, deleting the
hippocampus between trials.

**Change:** treat eviction as a bug class, not a policy. Implemented in
`pig-harness` Göktürk branch (`inference/agent/compaction.py`): dropped blocks
are distilled before deletion, never silently truncated.

**Measurement:** compaction-on vs compaction-off, ≥20 seeds (see §7).

## 2. Hippocampal replay and consolidation — what to do between levels

Sharp-wave-ripple replay compresses recent trajectories and reactivates them
offline, both awake-resting and in sleep. Functionally it does two things:
stabilises episodic traces into cortical structure, and supports planning by
replaying candidate trajectories forward.

Our agent currently does the opposite at exactly the wrong moment: on level
transition we *clear* the compaction store and the dead-signature store,
because the new level may have different rules.

That is half right. Layout and specific objects change; *mechanics* often
carry over — the ARC-AGI-3 environments are explicitly built so that later
levels compose earlier mechanics.

**Change:** split memory into two stores at level boundaries.
- *Episodic, level-scoped*: object positions, which cells were inert, current
  plan. Cleared on level change (as now).
- *Semantic, game-scoped*: "UP moves the avatar one cell", "orange tiles are
  walls", "clicking the counter does nothing". Survives level change.

Between levels, run one consolidation turn with no game actions: replay the
trajectory summary and ask the model to write the game-scoped rules it now
believes. Cost: one LLM call, zero game actions — and game actions are the
only thing the score charges for.

**Measurement:** L2 pass rate conditional on L1 pass. Duck's own numbers show
a hard depth wall here (`bp35`: 18/20 runs score, mean only 0.31 — it clears
L1 reliably and never reaches L3).

## 3. Perseveration — our no-op problem is a known frontal syndrome

Repeating an action that no longer produces the expected outcome, after the
rule has changed, is the classic signature of prefrontal damage in the
Wisconsin Card Sorting Test. Intact frontal cortex suppresses the prepotent
response once feedback contradicts it.

We measured exactly this: no-op rates of 30–75% per game (ft09: 112 of 150
actions produced no board change). The agent knew, mechanically, that those
clicks were inert — the `DeadSignatureStore` had recorded them — but that
knowledge was applied only to the deterministic fallback's candidate list, not
surfaced to the model. The model kept re-deriving the same wrong action.

**Change (done):** inert *object types* — position-independent
colour+size+shape signatures, not just coordinates — are now injected into the
prompt as an explicit "already tried, no effect" block. Plans abort on the
first no-op rather than the third.

**Measurement:** no-op fraction per game, before/after. This is a much lower
variance statistic than score and can be read off a single run.

## 4. Predictive coding — the signal we compute but do not use

The dominant framework for cortical function holds that the brain continuously
predicts its sensory input and propagates only the *error*. Learning is driven
by surprise; unsurprising input teaches nothing.

Our `render.object_diff` already computes a prediction-error-shaped signal
("colour=9 size=15 moved (r40,c13) → (r40,c14) [dr+0,dc+1]"). But nothing ever
issued a prediction, so the diff is only descriptive, never diagnostic. The
model cannot tell "the world did what I expected" from "the world surprised
me" — which is precisely the bit that should drive belief revision.

**Change:** require an explicit prediction before acting. The plan format
becomes `{action, expected_effect}`. After execution, the harness compares the
observed `object_diff` to the stated expectation and labels the result
CONFIRMED / VIOLATED. A violated prediction forces a re-plan; a long run of
confirmations licenses longer action batches without re-planning.

This is cheap (no extra LLM calls — the prediction rides along in the existing
plan) and it directly attacks the dominant failure mode: 49.6% of duck's runs
complete zero levels, which is the signature of an early wrong hypothesis that
never gets corrected.

**Measurement:** fraction of actions taken under a VIOLATED prediction; L1
pass rate.

## 5. Place cells, grid cells, path integration — the keyboard-game gap

The single clearest structural weakness in the reference harness is by control
modality:

| tag | games | mean score |
|---|---:|---:|
| `click` | 7 | 2.20 |
| `keyboard_click` | 13 | 1.06 |
| `keyboard` | 4 | **0.12** |

Two of the four keyboard games score zero across 20 attempts. Lifting the
keyboard group to the click average alone would move the total from 1.60 to
~1.93 — above the current leaderboard top.

The neuroscience is unusually on-point. Spatial navigation in mammals rests on
a dedicated substrate: place cells (position), grid cells (metric), head-
direction cells (heading), boundary-vector cells (walls). Critically, this is
an *allocentric map* supporting path integration and vector-based planning —
not a policy learned by trial and error.

Click games do not need this: the action space is "touch the thing", which is
an affordance judgement (dorsal-stream, parietal) and maps well onto our
connected-component candidate generator. Keyboard games need the map: an
avatar, a heading, walls, and a route.

The reference harness's prompt actively fights this, telling the model "do not
assume a player exists" and "do not frame the objective as reaching a specific
absolute row or column" — reasonable advice for logic puzzles, corrosive for
navigation, and contradicted by the same prompt's own advice to write BFS.

**Change:** modality-aware perception and prompting.
- If the action set contains no click, run an avatar-identification probe:
  press each direction once, find the component whose centroid translates
  consistently. That is 4 actions and yields the avatar, the metric (cells per
  press), and the direction mapping.
- Build an explicit occupancy grid (passable / blocked) from the components
  that did *not* move, and expose a `path_to(target)` BFS helper in the Python
  tool so the model plans routes instead of stepping blindly.
- Our harness already has a head start here: we use semantic action names
  UP/DOWN/LEFT/RIGHT (validated on 16 games: 38 agreements, 1 disagreement),
  where the reference uses opaque ACTION1..4.

**Measurement:** the 4 keyboard games as a clean subset — two are at a hard
zero in the baseline, so any level completion is signal that clears the noise
band.

## 6. Successor representation — why goal changes are cheap for brains

Hippocampal and orbitofrontal codes look less like cached values and more like
a *predictive map*: expected future state occupancy under the current policy
(the successor representation). Its practical virtue is that when the goal
moves but the transition structure does not, you re-plan by recombining the
existing map instead of relearning from scratch.

ARC-AGI-3 is built exactly this way: levels change the goal and the layout
while reusing mechanics.

**Change:** persist the *transition model* across levels even while clearing
episodic detail — "UP moves avatar −1 row unless blocked", "clicking type X
toggles colour". Our `GraphExplorer` already builds a state-transition graph;
today it is thrown away at level boundaries and never shown to the model. Make
its learned action-effect summary part of the game-scoped semantic store (§2).

## 7. Noise, and the honest limit of all of this

The reference harness's own 500-run analysis gives, for a single 25-game pass:
mean 1.600, std **0.448**, range 0.999–2.382. The gap between first and fourth
on the public leaderboard is inside one standard deviation of a single run.

Consequence: **no change in this document may be called an improvement on
fewer than ~20 seeds.** Two of my own earlier A/B runs (3 and 10 passes on 2
games) were under-powered by design and are void. A third was invalidated
outright by a configuration bug — a 10-minute per-game cap collapsed the
request timeout to 0.1 s, producing 5,788 timeout errors and a median of 35
actions per run.

## 8. Where the analogy stops

Stated plainly so it does not get over-extended:

- Artificial networks are not models of cortex. Backpropagation has no
  established biological implementation, and nothing in a transformer
  corresponds to dendritic computation, neuromodulation, or spike timing.
- "Attention" in transformers and attention in cognitive neuroscience are
  different things that share a name.
- Biological sample efficiency rests heavily on strong innate priors —
  objectness, agency, cause, number — accumulated over evolution. Chollet's
  "core knowledge priors" framing is the relevant bridge, and it argues for
  *engineering those priors into perception*, which is what our
  connected-component and affordance layer does, rather than expecting them to
  emerge.
- Everything above is a source of engineering hypotheses. Each one still has
  to survive §7.

## 9. Ranked action list

| # | Change | Grounding | Expected leverage | Cost |
|---|---|---|---|---|
| 1 | Modality-aware keyboard handling: avatar probe, occupancy grid, `path_to` BFS | §5 place/grid cells | Highest — 4 games at 0.12 vs 2.20; alone moves 1.60→~1.93 | Medium |
| 2 | Predict-then-verify plan format | §4 predictive coding | High — attacks the 49.6% zero-level runs | Low |
| 3 | Game-scoped semantic store + between-level consolidation turn | §2 replay, §6 SR | High — attacks the depth wall | Low |
| 4 | Compaction instead of eviction | §1 CLS | Proven elsewhere (13.3%→38.3%) | Done, unmeasured |
| 5 | Perseveration guard: inert types in prompt, abort plan on first no-op | §3 frontal control | Medium — no-op was 30–75% | Done |

## Sources

- ARC-AGI-3 technical report — https://arxiv.org/html/2603.24621v1
- Explore Before You Solve: speed–depth trade-off in epistemic agents for
  ARC-AGI-3 — https://arxiv.org/pdf/2605.25931
- Graph-based exploration for ARC-AGI-3 — https://arxiv.org/pdf/2512.24156
- Hippocampal-memory-inspired scalable RL —
  https://www.nature.com/articles/s41598-025-10586-x
- Hippocampal function and reinforcement learning (editorial) —
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12011858/
- When brain-inspired AI meets AGI — https://arxiv.org/pdf/2303.15935
- Neuroscience-inspired continuous learning systems —
  https://arxiv.org/html/2504.20109v1
- OpenAI, two settings tripled ARC-AGI-3 scores —
  https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/
