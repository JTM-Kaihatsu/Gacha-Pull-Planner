import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts'

// Colors per banner type (net pulls and refund shade)
const PHASE_COLORS = {
  char:   { net: '#7c3aed', refund: '#a78bfa' },
  weapon: { net: '#d97706', refund: '#fbbf24' },
}

function buildChartData(vizSample) {
  // Sort: successes ascending by pulls used, then failures descending
  const successes = [...vizSample.filter(r => r.success)].sort((a, b) => a.total_pulls_used - b.total_pulls_used)
  const failures  = [...vizSample.filter(r => !r.success)].sort((a, b) => b.total_pulls_used - a.total_pulls_used)
  const sorted = [...successes, ...failures]

  return sorted.map((run, idx) => {
    const obj = { runIdx: idx, success: run.success, _run: run }
    for (const phase of run.phases) {
      const net = Math.max(0, phase.pulls_used - phase.refunds)
      obj[`${phase.label}_net`]    = parseFloat(net.toFixed(1))
      obj[`${phase.label}_refund`] = parseFloat(phase.refunds.toFixed(1))
    }
    return obj
  })
}

function getPhaseKeys(vizSample) {
  if (!vizSample.length) return []
  const phases = vizSample[0].phases
  return phases.map(p => ({ label: p.label, banner: p.banner }))
}

const CustomTooltip = ({ active, payload, label, showRefunds = true }) => {
  if (!active || !payload?.length) return null
  const run = payload[0]?.payload?._run
  if (!run) return null

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs shadow-xl max-w-[200px]">
      <div className="text-slate-500 mb-1">Run #{run.trial?.toLocaleString()}</div>
      <div className={`font-semibold mb-2 ${run.success ? 'text-violet-300' : 'text-red-400'}`}>
        {run.success ? 'Success' : 'Failure'} — {run.total_pulls_used} pulls
      </div>
      {run.phases.map(p => (
        <div key={p.label} className="mb-1">
          <span className={`font-medium ${p.banner === 'char' ? 'text-violet-400' : 'text-amber-400'}`}>
            {p.label}
          </span>
          <span className="text-slate-300 ml-1">
            {p.pulls_used} pulls
          </span>
          {showRefunds && p.refunds > 0 && (
            <span className="text-slate-500 ml-1">
              ({p.refunds.toFixed(1)} refunds)
            </span>
          )}
        </div>
      ))}
    </div>
  )
}

export default function PullsChart({ vizSample, totalPulls, sampleSize = 500, showRefunds = true }) {
  if (!vizSample?.length) return null

  // Slice proportionally: keep success/failure ratio intact
  const totalRuns  = vizSample.length
  const ratio      = sampleSize / totalRuns
  const successes  = vizSample.filter(r => r.success)
  const failures   = vizSample.filter(r => !r.success)
  const sliced = [
    ...successes.slice(0, Math.round(successes.length * ratio)),
    ...failures.slice(0,  Math.round(failures.length  * ratio)),
  ]

  const data = buildChartData(sliced)
  const phaseKeys = getPhaseKeys(vizSample)
  const successCount = sliced.filter(r => r.success).length
  const failCount    = sliced.filter(r => !r.success).length

  return (
    <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-4">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-slate-300">Pull Distribution by Run</span>
        <span className="text-xs text-slate-500">{vizSample.length} sampled runs</span>
      </div>
      <div className="flex gap-4 mb-3 text-xs text-slate-400">
        <span><span className="inline-block w-2 h-2 rounded-sm bg-violet-600 mr-1" />Char pulls</span>
        {showRefunds && <span><span className="inline-block w-2 h-2 rounded-sm bg-violet-400 mr-1" />Char refunds</span>}
        <span><span className="inline-block w-2 h-2 rounded-sm bg-amber-600 mr-1" />Weapon pulls</span>
        {showRefunds && <span><span className="inline-block w-2 h-2 rounded-sm bg-amber-400 mr-1" />Weapon refunds</span>}
        <span><span className="inline-block w-3 h-2 border border-red-500 mr-1" />Failed run</span>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} barCategoryGap={1} barGap={0}>
          <XAxis dataKey="runIdx" hide />
          <YAxis
            domain={[0, totalPulls]}
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={32}
          />
          <Tooltip content={<CustomTooltip showRefunds={showRefunds} />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <ReferenceLine
            y={totalPulls}
            stroke="#ef4444"
            strokeDasharray="4 3"
            strokeOpacity={0.5}
          />

          {phaseKeys.flatMap(({ label, banner }) => [
            <Bar
              key={`${label}_net`}
              dataKey={`${label}_net`}
              stackId="run"
              fill={PHASE_COLORS[banner].net}
              isAnimationActive={false}
            >
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={PHASE_COLORS[banner].net}
                  stroke={entry.success ? 'none' : '#ef4444'}
                  strokeWidth={entry.success ? 0 : 1}
                  opacity={entry.success ? 1 : 0.65}
                />
              ))}
            </Bar>,
            <Bar
              key={`${label}_refund`}
              dataKey={`${label}_refund`}
              stackId="run"
              fill={PHASE_COLORS[banner].refund}
              isAnimationActive={false}
            >
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={PHASE_COLORS[banner].refund}
                  stroke={entry.success ? 'none' : '#ef4444'}
                  strokeWidth={entry.success ? 0 : 1}
                  opacity={entry.success ? 0.85 : 0.5}
                />
              ))}
            </Bar>,
          ])}
        </BarChart>
      </ResponsiveContainer>

      <div className="flex justify-between text-xs text-slate-500 mt-1 px-8">
        <span>← {successCount} successes</span>
        <span>{failCount} failures →</span>
      </div>
    </div>
  )
}
