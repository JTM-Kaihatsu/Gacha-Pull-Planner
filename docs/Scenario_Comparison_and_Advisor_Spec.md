# Scenario Comparison and Advisor Spec

Design for replacing the current one-shot LLM verdict with three layers: a
deterministic baseline read, one-click scenario comparisons, and an optional
open-ended advisor. The guiding principle is deterministic first, with AI
reserved only for questions that cannot be answered by a preset.

## Motivation

The current AI verdict does two things, and both are handled better without a model:

1. It restates the numbers the user can already see. That is presentation, not
   insight, and it can be templated.
2. It proposes a seemingly arbitrary number of extra pulls. That number can be
   computed exactly by re-running the simulation, and framing it as "spend more"
   reads like a nudge to spend.

So the redesign removes the one-shot verdict and splits its value across the three
layers below. A model only earns its place where a preset cannot go: translating
open-ended natural language into a simulation run and interpreting the result.

## Layer 1: Deterministic baseline read (replaces the current verdict)

A pure function that turns the stats dict into a short, structured read. No network
call, no cost, always shown.

Inputs: the existing `stats_summary` (success rate, leftover pulls, most common
failure state, correlation stats, starting guarantees).

Outputs (all derived by thresholds, no model):
1. A confidence label bucketed from success rate: comfortable, tight, or a stretch.
2. The margin note: average leftover pulls on success, so the user sees how much
   room they have.
3. The key lever: read from the correlation stats which factor decides the outcome
   (for example, "you fail almost entirely on the second 50/50, not on pull count").
4. Honest framing that sometimes recommends not spending. Example: "You are
   comfortably there. More pulls will not change much." Example: "This is a coin
   flip. If you cannot accept the loss, wait for a guarantee or the rerun rather
   than adding pulls."

For "how many more pulls would help," compute it, do not guess it. Run the
simulation at increasing `total_pulls` until a target success rate is reached (for
example 80 percent). Present the result neutrally next to the alternative of waiting
for a guarantee. This is Scenario A (below) run automatically.

## Layer 2: Deterministic scenario comparison (the core new feature, implemented)

The user has a baseline result. They pick a preset what-if. The backend re-runs the
simulation with a transformed request. The UI shows baseline versus new side by
side, with computed deltas and a one-line templated interpretation. The user keeps
their baseline instead of losing it by editing the form.

### Why this is not an AI feature

Every preset is a fixed parameter transform with clear eligibility rules. A preset
that has preconditions is a button, not a conversation. Buttons are instant, free,
cannot hallucinate, and cannot misread intent.

### Presets

Let `C` be the number of character copies in the goal and `W` the number of weapon
copies.

| ID | Label | Transform | Eligibility |
|----|-------|-----------|-------------|
| A | What if I get X more pulls | `total_pulls += X` (stepper, for example plus 10, 20, 30) | Always |
| B | What if I cap my budget at X | `total_pulls = X`, where X is below the current value | Current `total_pulls` is above the minimum needed to attempt the goal |
| C | What if I drop a copy | Remove one character copy or one weapon copy from the strategy | `C + W > 1`. The user picks which to decrement. Character is selectable only when `C > 0`, weapon only when `W > 0`, and the result must leave at least one copy |
| D | What if I change the order | Reorder the strategy phases (for example pull the weapon earlier or later) | `C > 1` and `W >= 1`, since order only matters with at least two characters and a weapon |

### API

Reuse the existing `POST /analyze` endpoint. The frontend already holds the baseline
result. It builds the transformed request, calls `/analyze` again, and computes the
delta in the UI. No new backend endpoint is required.

### Delta presentation

Show baseline versus new for at least: success rate, average leftover pulls on
success, and the most common failure state. Add a one-line templated interpretation
keyed off the success rate delta:
- Large positive delta: "This helps a lot. The extra pulls let you absorb a lost
  50/50."
- Small delta: "This barely moves the odds. Your bottleneck is winning the coin
  flips, not pull count."
- Negative delta: state the cost plainly.

## Layer 3: Optional open-ended advisor (the only real AI use, implemented)

A free-text box for questions that are not covered by the presets. The model maps an
open-ended question to a simulation call using tool use, then interprets the result
against the baseline. This is the one thing a template cannot do: understand
unbounded intent and drive the simulator from it.

Design:
- Define a tool that runs the simulation with the same parameters as `/analyze`.
- Give the model the baseline scenario plus the user question. It decides the
  parameter changes, calls the tool, reads the result, and writes a short comparison.
- Guardrails: validate parameters before running, cap the number of tool calls per
  question, keep the OpenAI spend cap, and keep it opt-in and off by default for
  cost. Reuse the graceful fallback already in place so a rate limit or error shows
  an honest message rather than failing the request.

Anti-slop measures (so a free-text box does not become chat slop):
- Context-aware suggested questions, chosen by the baseline confidence, that populate
  the input (they do not auto-send). They steer toward open-ended questions the
  presets do not cover.
- Grounded output: the prompt requires the model to cite the specific success rates
  it got from the tool, and the runs it made are returned and shown as small receipts
  above the answer (for example "220 pulls, 1C/1W to 71%"), so the answer reads as
  analysis, not vibes.

If the preset set (Layer 2) turns out to cover the common questions, Layer 3 stays a
cherry on top rather than a core dependency. That is an acceptable and honest outcome.

## What is removed

The current one-shot `analyze_sim_result` verdict as the primary AI feature. Its
factual content moves to Layer 1. Its "add pulls" suggestion becomes a deterministic
computation in Layer 1. Its open-ended value moves to Layer 3.

## Ethics

Advice should sometimes recommend not spending. Pull deltas are framed neutrally, not
as encouragement to spend. No dark patterns.

## Rough effort

- Layer 1, deterministic read: small to medium. A pure function plus tests, wired as
  the default output.
- Layer 2, scenario comparison: medium. A frontend panel, reuse of `/analyze`, delta
  UI, and templated interpretation.
- Layer 3, open-ended advisor: medium. A tool-calling loop, a small UI, and
  guardrails. Optional, and best done last.
