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
    // Scenario comparisons are a quick, deterministic aside; never spend the
    // AI budget on them regardless of the main form's toggle.
    enable_ai_analysis: false,
    char_pity_config: baseline.charPityConfig,
    weapon_pity_config: baseline.weaponPityConfig,
  }
}

const HELP_THRESHOLD = { big: 10, small: 5 }

export function compareScenarios(baselineStats, newStats) {
  const basePct = parsePercent(baselineStats.success_rate)
  const newPct  = parsePercent(newStats.success_rate)
  const delta   = newPct - basePct

  let interpretation
  if (delta >= HELP_THRESHOLD.big) {
    interpretation = 'This helps a lot. The extra room lets you absorb a lost coin flip.'
  } else if (delta <= -HELP_THRESHOLD.big) {
    interpretation = 'This costs you a real chunk of your odds.'
  } else if (Math.abs(delta) < HELP_THRESHOLD.small) {
    interpretation = 'This barely moves the odds. Your bottleneck is winning the coin flips, not pull count.'
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
