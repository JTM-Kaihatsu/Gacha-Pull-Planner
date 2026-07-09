import { useState } from 'react'
import { analyze } from '../api'
import { buildOrderingOptions } from './StrategyBuilder'
import { buildScenarioPayload, compareScenarios, isValidPullDelta } from '../lib/scenarios'

const NEGATIVE_CHIPS = [-30, -20, -10]
const POSITIVE_CHIPS = [10, 20, 30]

export default function ScenarioComparison({ baseline, baselineStats }) {
  const [comparison, setComparison] = useState(null)
  const [activeLabel, setActiveLabel] = useState(null)
  const [loadingLabel, setLoadingLabel] = useState(null)
  const [scenarioError, setScenarioError] = useState(null)
  const [pullDeltaInput, setPullDeltaInput] = useState('0')

  const { desiredChars: C, desiredWeapons: W, weaponAfter } = baseline
  const totalPulls = baseline.form.total_pulls

  const showDrop = (C + W) > 1
  const showOrder = C > 1 && W >= 1
  const orderOptions = showOrder
    ? buildOrderingOptions(C).filter(opt => opt.value !== weaponAfter)
    : []

  const pullDelta = parseInt(pullDeltaInput, 10) || 0
  const newTotalPulls = totalPulls + pullDelta
  const pullDeltaValid = isValidPullDelta(totalPulls, pullDelta)
  const pullsSubmitting = loadingLabel === 'pulls'

  async function runScenario(label, overrides) {
    setLoadingLabel(label)
    setScenarioError(null)
    try {
      const payload = buildScenarioPayload(baseline, overrides)
      const data = await analyze(payload)
      setComparison({
        ...compareScenarios(baselineStats, data.stats_summary),
        baseTotalPulls: totalPulls,
        newTotalPulls: payload.total_pulls,
      })
      setActiveLabel(label)
    } catch (err) {
      setScenarioError(err.message)
    } finally {
      setLoadingLabel(null)
    }
  }

  function adjustPullDelta(amount) {
    setPullDeltaInput(String(pullDelta + amount))
  }

  function handleResubmitPulls() {
    if (!pullDeltaValid) return
    runScenario('pulls', { total_pulls: newTotalPulls })
  }

  function clear() {
    setComparison(null)
    setActiveLabel(null)
    setScenarioError(null)
  }

  return (
    <details className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 cursor-pointer">
      <summary className="text-sm font-medium text-slate-400 select-none">Compare Scenarios</summary>
      <div className="mt-4 space-y-5">
        <p className="text-xs text-slate-500">
          Try a quick what-if against your baseline result. These re-run the
          simulation and never use the optional AI verdict.
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
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <span className="text-xs text-slate-500">
              {newTotalPulls < 1 ? (
                <span className="text-red-400">Resulting total must be at least 1 pull.</span>
              ) : (
                <>New total: <span className="text-slate-300 font-medium">{newTotalPulls}</span> pulls (baseline: {totalPulls})</>
              )}
            </span>
            <button
              type="button"
              onClick={handleResubmitPulls}
              disabled={!pullDeltaValid || pullsSubmitting}
              className="text-xs font-medium px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
            >
              {pullsSubmitting ? 'Running…' : 'Resubmit and Compare Results'}
            </button>
          </div>
        </ScenarioGroup>

        {showDrop && (
          <ScenarioGroup label="What if I drop a copy?">
            <ChipButton
              disabled={C === 0}
              loading={loadingLabel === 'drop-char'}
              active={activeLabel === 'drop-char'}
              onClick={() => runScenario('drop-char', { desiredChars: C - 1 })}
            >
              Drop 1 character copy
            </ChipButton>
            <ChipButton
              disabled={W === 0}
              loading={loadingLabel === 'drop-weapon'}
              active={activeLabel === 'drop-weapon'}
              onClick={() => runScenario('drop-weapon', { desiredWeapons: W - 1 })}
            >
              Drop 1 weapon copy
            </ChipButton>
          </ScenarioGroup>
        )}

        {showOrder && orderOptions.length > 0 && (
          <ScenarioGroup label="What if I change the pull order?">
            {orderOptions.map(opt => (
              <ChipButton
                key={opt.value}
                loading={loadingLabel === `order-${opt.value}`}
                active={activeLabel === `order-${opt.value}`}
                onClick={() => runScenario(`order-${opt.value}`, { weaponAfter: opt.value })}
              >
                {opt.label}
              </ChipButton>
            ))}
          </ScenarioGroup>
        )}

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

function ChipButton({ children, onClick, active, loading, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || loading}
      className={`text-xs font-medium px-3 py-1.5 rounded-lg border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
        active
          ? 'bg-violet-600 border-violet-500 text-white'
          : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-violet-500'
      }`}
    >
      {loading ? 'Running…' : children}
    </button>
  )
}

function ComparisonCard({ comparison, onClear }) {
  const {
    basePct, newPct, delta, baseLeftover, newLeftover, baseFailure, newFailure,
    interpretation, baseTotalPulls, newTotalPulls,
  } = comparison
  const deltaColor = delta > 0 ? 'text-emerald-400' : delta < 0 ? 'text-red-400' : 'text-slate-400'
  const sign = delta > 0 ? '+' : ''

  return (
    <div className="bg-slate-900/60 border border-slate-700 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Baseline vs Scenario</span>
        <button type="button" onClick={onClear} className="text-xs text-slate-500 hover:text-slate-300">
          Clear
        </button>
      </div>

      <div className="text-sm text-slate-300">
        <span className="text-xs text-slate-500 uppercase tracking-wider block mb-0.5">Total Pulls Simulated On</span>
        {baseTotalPulls} <span className="text-slate-600">to</span>{' '}
        <span className="font-semibold text-white">{newTotalPulls}</span>
        {baseTotalPulls === newTotalPulls && (
          <span className="text-slate-600 ml-1">(unchanged)</span>
        )}
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
