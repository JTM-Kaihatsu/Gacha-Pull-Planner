// Pure helpers for the deterministic scenario-comparison panel (Layer 2 of the
// advisor redesign, see docs/Scenario_Comparison_and_Advisor_Spec.md).
//
// Every scenario here is a fixed parameter transform with clear eligibility
// rules, so this stays plain arithmetic and template strings. No AI, no cost.
import { buildStrategy } from '../components/StrategyBuilder'

export function parsePercent(value) {
  if (typeof value === 'string') {
    const n = parseFloat(value.replace('%', ''))
    return Number.isNaN(n) ? 0 : n
  }
  return typeof value === 'number' ? (value <= 1 ? value * 100 : value) : 0
}

// Builds a full /analyze payload for a modified scenario. `overrides` may set
// total_pulls, desiredChars, desiredWeapons, and/or weaponAfter; anything not
// overridden falls back to the baseline snapshot taken at the last Simulate.
export function buildScenarioPayload(baseline, overrides = {}) {
  const desiredChars   = overrides.desiredChars   ?? baseline.desiredChars
  const desiredWeapons = overrides.desiredWeapons ?? baseline.desiredWeapons
  const showOrdering   = desiredChars > 1 && desiredWeapons >= 1
  const weaponAfter    = showOrdering
    ? Math.min(overrides.weaponAfter ?? baseline.weaponAfter, desiredChars)
    : desiredChars

  const strategy = buildStrategy(desiredChars, desiredWeapons, weaponAfter)

  return {
    ...baseline.form,
    total_pulls: overrides.total_pulls ?? baseline.form.total_pulls,
    strategy,
    full_4star_chars: baseline.full4StarChars,
    char_pity_config: baseline.charPityConfig,
    weapon_pity_config: baseline.weaponPityConfig,
  }
}

// A pull-budget delta is only submittable if it actually changes something and
// does not push the total below 1.
export function isValidPullDelta(baselineTotalPulls, delta) {
  return delta !== 0 && baselineTotalPulls + delta >= 1
}

// Context-aware starter questions for the follow-up advisor, chosen by how the
// baseline turned out (its confidence) so they steer toward useful, open-ended
// questions the presets do not already answer. One general question is always
// included. These populate the input; they do not auto-send.
export function suggestedQuestions(confidence) {
  let bucket
  if (confidence === 'comfortable' || confidence === 'likely') {
    bucket = [
      'Can I afford to add another character copy and still stay safe?',
      'How many pulls could I cut and still be comfortable?',
    ]
  } else if (confidence === 'coin_flip') {
    bucket = [
      'If I lose my first 50/50, is it still worth continuing?',
      'Is it worth waiting for the rerun or aiming for a later character banner instead of gambling now?',
    ]
  } else {
    // stretch, long_shot, or unknown: triage questions
    bucket = [
      'Assuming my budget is fixed, how should I adjust my character and weapon copy strategy?',
      'How many more pulls would I realistically need to feel safe?',
    ]
  }
  return [...bucket, 'Why do I usually fail here, and is it the coin flips or the pull count?']
}

const HELP_THRESHOLD = { big: 10, small: 5 }

export function compareScenarios(baselineStats, newStats) {
  const basePct = parsePercent(baselineStats.success_rate)
  const newPct  = parsePercent(newStats.success_rate)
  const delta   = newPct - basePct

  let interpretation
  if (delta >= HELP_THRESHOLD.big) {
    interpretation = 'This meaningfully improves your odds.'
  } else if (delta <= -HELP_THRESHOLD.big) {
    interpretation = 'This costs you a real chunk of your odds.'
  } else if (Math.abs(delta) < HELP_THRESHOLD.small) {
    interpretation = 'This barely moves the odds.'
  } else if (delta > 0) {
    interpretation = 'This helps some, but is not a decisive swing on its own.'
  } else {
    interpretation = 'This costs you some odds, but is not a decisive swing on its own.'
  }

  return {
    basePct,
    newPct,
    delta,
    baseLeftover: baselineStats.avg_leftover_pulls_on_success,
    newLeftover: newStats.avg_leftover_pulls_on_success,
    baseFailure: baselineStats.most_common_failure_state,
    newFailure: newStats.most_common_failure_state,
    interpretation,
  }
}
