import { useState } from 'react'
import { analyze } from '../api'
import { StrategyPreview } from './StrategyBuilder'
import { buildScenarioPayload, compareScenarios } from '../lib/scenarios'

const NEGATIVE_CHIPS = [-30, -20, -10]
const POSITIVE_CHIPS = [10, 20, 30]
const MAX_CHARS = 7
const MAX_WEAPONS = 5

const pluralize = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`

export default function ScenarioComparison({ baseline, baselineStats }) {
  const totalPulls = baseline.form.total_pulls
  const baseChars = baseline.desiredChars
  const baseWeapons = baseline.desiredWeapons

  const [pullDeltaInput, setPullDeltaInput] = useState('0')
  const [newChars, setNewChars] = useState(baseChars)
  const [newWeapons, setNewWeapons] = useState(baseWeapons)
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [scenarioError, setScenarioError] = useState(null)

  const pullDelta = parseInt(pullDeltaInput, 10) || 0
  const newTotalPulls = totalPulls + pullDelta

  const showOrdering = newChars > 1 && newWeapons >= 1
  const previewWeaponAfter = Math.min(baseline.weaponAfter, newChars)

  const totalValid = newTotalPulls >= 1
  const goalValid = (newChars + newWeapons) >= 1
  const changed = pullDelta !== 0 || newChars !== baseChars || newWeapons !== baseWeapons
  const canCompare = totalValid && goalValid && changed && !loading

  function adjustPullDelta(amount) {
    setPullDeltaInput(String(pullDelta + amount))
  }

  async function handleCompare() {
    if (!canCompare) return
    setLoading(true)
    setScenarioError(null)
    try {
      const payload = buildScenarioPayload(baseline, {
        total_pulls: newTotalPulls,
        desiredChars: newChars,
        desiredWeapons: newWeapons,
      })
      const data = await analyze(payload)
      setComparison({
        ...compareScenarios(baselineStats, data.stats_summary),
        baseTotalPulls: totalPulls,
        newTotalPulls: payload.total_pulls,
        baseChars, baseWeapons, newChars, newWeapons,
      })
    } catch (err) {
      setScenarioError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function clear() {
    setComparison(null)
    setScenarioError(null)
  }

  return (
    <details className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 cursor-pointer">
      <summary className="text-sm font-medium text-slate-400 select-none">Compare Scenarios</summary>
      <div className="mt-4 space-y-5">
        <p className="text-xs text-slate-500">
          Build a what-if against your baseline result, then compare. This re-runs
          the simulation instantly and uses no AI.
        </p>

        <ScenarioGroup label="What if I changed my pull budget?">
          <div className="flex flex-wrap items-center gap-2">
            {NEGATIVE_CHIPS.map(n => (
              <ChipButton key={n} onClick={() => adjustPullDelta(n)}>{n}</ChipButton>
            ))}
            <input
              type="number"
              value={pullDeltaInput}
              onChange={e => setPullDeltaInput(e.target.value)}
              className="w-20 text-center bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-white focus:outline-none focus:border-violet-500"
            />
            {POSITIVE_CHIPS.map(n => (
              <ChipButton key={n} onClick={() => adjustPullDelta(n)}>+{n}</ChipButton>
            ))}
          </div>
          <div className="text-xs">
            {totalValid ? (
              <span className="text-slate-500">
                New total: <span className="text-slate-300 font-medium">{newTotalPulls}</span> pulls (baseline: {totalPulls})
              </span>
            ) : (
              <span className="text-red-400">Resulting total must be at least 1 pull.</span>
            )}
          </div>
        </ScenarioGroup>

        <ScenarioGroup label="What if I changed my goal?">
          <div className="space-y-2">
            <Stepper
              label="Character copies"
              onDecrement={() => setNewChars(v => Math.max(0, v - 1))}
              onIncrement={() => setNewChars(v => Math.min(MAX_CHARS, v + 1))}
              decDisabled={newChars <= 0}
              incDisabled={newChars >= MAX_CHARS}
            />
            <Stepper
              label="Weapon copies"
              onDecrement={() => setNewWeapons(v => Math.max(0, v - 1))}
              onIncrement={() => setNewWeapons(v => Math.min(MAX_WEAPONS, v + 1))}
              decDisabled={newWeapons <= 0}
              incDisabled={newWeapons >= MAX_WEAPONS}
            />
          </div>

          <div className="text-xs text-slate-400">
            New goal: {pluralize(newChars, 'character')} and {pluralize(newWeapons, 'weapon')}
          </div>

          {goalValid ? (
            <StrategyPreview
              desiredChars={newChars}
              desiredWeapons={newWeapons}
              weaponAfter={previewWeaponAfter}
              showOrdering={showOrdering}
            />
          ) : (
            <p className="text-xs text-red-400">Pick at least one character or weapon copy.</p>
          )}
        </ScenarioGroup>

        <div className="flex justify-end">
          <button
            type="button"
            onClick={handleCompare}
            disabled={!canCompare}
            className="text-sm font-medium px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
          >
            {loading ? 'Running…' : 'Compare Simulations'}
          </button>
        </div>

        {scenarioError && <p className="text-red-400 text-sm">{scenarioError}</p>}

        {comparison && <ComparisonCard comparison={comparison} onClear={clear} />}
      </div>
    </details>
  )
}

function ScenarioGroup({ label, children }) {
  return (
    <div className="space-y-2">
      <div className="text-xs text-slate-500 uppercase tracking-wider">{label}</div>
      {children}
    </div>
  )
}

function ChipButton({ children, onClick, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="text-xs font-medium px-3 py-1.5 rounded-lg border bg-slate-800 border-slate-700 text-slate-300 hover:border-violet-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
    >
      {children}
    </button>
  )
}

function Stepper({ label, onDecrement, onIncrement, decDisabled, incDisabled }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-slate-300 w-32">{label}</span>
      <StepButton onClick={onDecrement} disabled={decDisabled}>−</StepButton>
      <StepButton onClick={onIncrement} disabled={incDisabled}>+</StepButton>
    </div>
  )
}

function StepButton({ children, onClick, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="w-8 h-8 flex items-center justify-center rounded-lg border bg-slate-800 border-slate-700 text-slate-300 text-lg leading-none hover:border-violet-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
    >
      {children}
    </button>
  )
}

function ComparisonCard({ comparison, onClear }) {
  const {
    basePct, newPct, delta, baseLeftover, newLeftover, baseFailure, newFailure,
    interpretation, baseTotalPulls, newTotalPulls, baseChars, baseWeapons, newChars, newWeapons,
  } = comparison
  const deltaColor = delta > 0 ? 'text-emerald-400' : delta < 0 ? 'text-red-400' : 'text-slate-400'
  const sign = delta > 0 ? '+' : ''
  const pullsChanged = baseTotalPulls !== newTotalPulls
  const goalChanged = baseChars !== newChars || baseWeapons !== newWeapons

  return (
    <div className="bg-slate-900/60 border border-slate-700 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Baseline vs Scenario</span>
        <button type="button" onClick={onClear} className="text-xs text-slate-500 hover:text-slate-300">
          Clear
        </button>
      </div>

      <div className="space-y-1">
        <div className="text-sm text-slate-300">
          <span className="text-xs text-slate-500 uppercase tracking-wider">Pulls simulated on: </span>
          {baseTotalPulls} <span className="text-slate-600">to</span>{' '}
          <span className="font-semibold text-white">{newTotalPulls}</span>
          {!pullsChanged && <span className="text-slate-600 ml-1">(unchanged)</span>}
        </div>
        <div className="text-sm text-slate-300">
          <span className="text-xs text-slate-500 uppercase tracking-wider">Goal: </span>
          {baseChars}C / {baseWeapons}W <span className="text-slate-600">to</span>{' '}
          <span className="font-semibold text-white">{newChars}C / {newWeapons}W</span>
          {!goalChanged && <span className="text-slate-600 ml-1">(unchanged)</span>}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-xs text-slate-500">Success Rate</div>
          <div className="text-sm text-slate-300">
            {basePct.toFixed(2)}% <span className="text-slate-600">to</span>{' '}
            <span className="font-semibold text-white">{newPct.toFixed(2)}%</span>
          </div>
          <div className={`text-xs font-medium ${deltaColor}`}>{sign}{delta.toFixed(2)} pts</div>
        </div>
        <div>
          <div className="text-xs text-slate-500">Leftover Pulls (success)</div>
          <div className="text-sm text-slate-300">
            {baseLeftover ?? 'n/a'} <span className="text-slate-600">to</span>{' '}
            <span className="font-semibold text-white">{newLeftover ?? 'n/a'}</span>
          </div>
        </div>
      </div>

      {(baseFailure || newFailure) && (
        <div className="text-xs text-slate-400">
          Most common failure: {baseFailure ? `${baseFailure.chars}C/${baseFailure.weapons}W` : 'none'}{' '}
          <span className="text-slate-600">to</span>{' '}
          {newFailure ? `${newFailure.chars}C/${newFailure.weapons}W` : 'none'}
        </div>
      )}

      <p className="text-sm text-slate-300 leading-relaxed">{interpretation}</p>
    </div>
  )
}
