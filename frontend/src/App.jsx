import { useState } from 'react'
import { analyze } from './api'
import PityInputs from './components/PityInputs'
import StrategyBuilder, { buildStrategy } from './components/StrategyBuilder'
import StatCard from './components/StatCard'
import SummaryBlock from './components/SummaryBlock'
import PullsChart from './components/PullsChart'
import ScenarioComparison from './components/ScenarioComparison'
import FollowUpAdvisor from './components/FollowUpAdvisor'
import AdvancedSettings, { CHAR_DEFAULTS, WEAPON_DEFAULTS } from './components/AdvancedSettings'
import './index.css'

const DEFAULT_FORM = {
  total_pulls: 180,
  start_char_pity: 0,
  start_char_guarantee: false,
  start_weapon_pity: 0,
  start_weapon_guarantee: false,
}

export default function App() {
  const [form, setForm]                 = useState(DEFAULT_FORM)
  const [desiredChars, setDesiredChars] = useState(1)
  const [desiredWeapons, setDesiredWeapons]     = useState(1)
  const [weaponAfter, setWeaponAfter]           = useState(1)
  const [loading, setLoading]           = useState(false)
  const [result, setResult]             = useState(null)
  const [baseline, setBaseline]         = useState(null)
  const [error, setError]               = useState(null)
  const [strategyError, setStrategyError]   = useState(null)
  const [sampleSize, setSampleSize]         = useState(500)
  const [full4StarChars, setFull4StarChars] = useState(true)
  const [charPityConfig, setCharPityConfig] = useState({ ...CHAR_DEFAULTS })
  const [weaponPityConfig, setWeaponPityConfig]     = useState({ ...WEAPON_DEFAULTS })

  function handleFormChange(key, value) {
    setForm(f => ({ ...f, [key]: value }))
  }

  function handleStrategyChange(key, value) {
    if (key === 'desiredChars') {
      setDesiredChars(value)
      setWeaponAfter(prev => Math.min(prev, value))
    } else if (key === 'desiredWeapons') {
      setDesiredWeapons(value)
    } else if (key === 'weaponAfter') {
      setWeaponAfter(value)
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (desiredChars === 0 && desiredWeapons === 0) {
      setStrategyError('Please select at least one character copy or one Weapon.')
      return
    }
    setStrategyError(null)
    setLoading(true)
    setError(null)
    setResult(null)
    setBaseline(null)

    const showOrdering = desiredChars > 1 && desiredWeapons >= 1
    const strategy = buildStrategy(desiredChars, desiredWeapons, showOrdering ? weaponAfter : desiredChars)

    // Snapshot of exactly what produced this result, so the scenario-comparison
    // panel can re-run variations without depending on live-editable form state.
    const snapshot = { form, desiredChars, desiredWeapons, weaponAfter, full4StarChars, charPityConfig, weaponPityConfig }

    try {
      const data = await analyze({
        ...form,
        strategy,
        full_4star_chars: full4StarChars,
        char_pity_config: charPityConfig,
        weapon_pity_config: weaponPityConfig,
      })
      setResult(data)
      setBaseline(snapshot)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const stats = result?.stats_summary

  return (
    <div className="min-h-screen bg-[#0f0f13] text-slate-200">
      <div className="max-w-2xl mx-auto px-4 py-10">

        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-white tracking-tight">Gacha Pull Simulator</h1>
          <p className="text-slate-400 mt-1 text-sm">Monte Carlo simulation for gacha banner planning</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6 bg-slate-900 border border-slate-800 rounded-2xl p-6">

          <div>
            <label className="block text-sm text-slate-400 mb-1">Total Pulls Available</label>
            <input
              type="number" min={1}
              value={form.total_pulls}
              onChange={e => handleFormChange('total_pulls', e.target.value === '' ? '' : +e.target.value)}
              onBlur={e => handleFormChange('total_pulls', Math.max(1, +e.target.value || 1))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-violet-500"
            />
          </div>

          <div>
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Current Pity</h2>
            <PityInputs form={form} onChange={handleFormChange} />
          </div>

          <label className="flex items-center gap-3 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={full4StarChars}
              onChange={e => setFull4StarChars(e.target.checked)}
              className="accent-violet-500 w-4 h-4"
            />
            <span className="text-sm text-slate-300">
              All 4★ characters on this banner are at max copies
              <span className="block text-xs text-slate-500 font-normal">Enable if you have max copies; duplicate pulls become refunded pulls</span>
            </span>
          </label>

          <AdvancedSettings
            charConfig={charPityConfig}
            weaponConfig={weaponPityConfig}
            onCharChange={setCharPityConfig}
            onWeaponChange={setWeaponPityConfig}
          />

          <div>
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Pull Goal & Strategy</h2>
            <StrategyBuilder
              desiredChars={desiredChars}
              desiredWeapons={desiredWeapons}
              weaponAfter={weaponAfter}
              onChange={handleStrategyChange}
              validationError={strategyError}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-xl font-semibold text-white bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Running 10,000 simulations…' : 'Simulate Pulls'}
          </button>

          {error && <p className="text-red-400 text-sm text-center">{error}</p>}
        </form>

        {result && stats && (
          <div className="mt-8 space-y-6">
            <h2 className="text-lg font-semibold text-white">Results for Successful Runs</h2>

            {result.summary && <SummaryBlock summary={result.summary} />}

            <div className="grid grid-cols-2 gap-3">
              <StatCard label="Success Rate" value={stats.success_rate} highlight />
              <StatCard label="Leftover Pulls (success)" value={stats.avg_leftover_pulls_on_success} />
              <StatCard label="Avg Char Pity" value={stats.avg_pity_char} />
              <StatCard label="Avg Weapon Pity" value={stats.avg_pity_weapon} />
              <StatCard label="Char 50/50 Win Rate" value={stats.successes_char_win_rate} />
              <StatCard label="Weapon 75/25 Win Rate" value={stats.successes_weapon_win_rate} />
            </div>

            {stats.viz_sample?.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-slate-400">Runs displayed: <span className="text-slate-200 font-medium">{sampleSize}</span></span>
                  <input
                    type="range" min={100} max={1000} step={50}
                    value={sampleSize}
                    onChange={e => setSampleSize(+e.target.value)}
                    className="w-40 accent-violet-500"
                  />
                </div>
                <PullsChart
                  vizSample={stats.viz_sample}
                  totalPulls={form.total_pulls}
                  sampleSize={sampleSize}
                  showRefunds={full4StarChars}
                />
              </div>
            )}

            {baseline && <ScenarioComparison baseline={baseline} baselineStats={stats} />}

            {baseline && <FollowUpAdvisor baseline={baseline} confidence={result.summary?.confidence} />}

            <details className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 cursor-pointer">
              <summary className="text-sm font-medium text-slate-400 select-none">Failed Run Stats</summary>
              <div className="mt-3 space-y-4">

                {stats.most_common_failure_state && (
                  <div className="bg-red-950/40 border border-red-800/50 rounded-lg px-4 py-3">
                    <div className="text-xs text-red-400 uppercase tracking-wider mb-1">Most Common Failure</div>
                    <div className="text-sm text-red-200">
                      Ran out after <span className="font-semibold text-white">{stats.most_common_failure_state.chars} char {stats.most_common_failure_state.chars === 1 ? 'copy' : 'copies'}</span> and <span className="font-semibold text-white">{stats.most_common_failure_state.weapons} weapon {stats.most_common_failure_state.weapons === 1 ? 'copy' : 'copies'}</span>
                      <span className="text-red-400 ml-2">({stats.most_common_failure_state.pct}% of failures)</span>
                    </div>
                  </div>
                )}

                {stats.failure_state_distribution?.length > 1 && (
                  <div>
                    <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Failure Breakdown</div>
                    <div className="space-y-1">
                      {stats.failure_state_distribution.map((s, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <div
                            className="h-2 rounded-full bg-red-700/60"
                            style={{ width: `${s.pct}%`, minWidth: '4px', maxWidth: '100%' }}
                          />
                          <span className="text-xs text-slate-400 whitespace-nowrap">
                            {s.chars}C / {s.weapons}W — {s.pct}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3">
                  <StatCard label="Char Win Rate (fails)" value={stats.failure_char_win_rate} />
                  <StatCard label="Weapon Win Rate (fails)" value={stats.failure_weapon_win_rate} />
                  <StatCard label="Leftover Pulls (fail)" value={stats.avg_leftover_pulls_on_failure} />
                </div>
              </div>
            </details>
          </div>
        )}
      </div>
    </div>
  )
}
