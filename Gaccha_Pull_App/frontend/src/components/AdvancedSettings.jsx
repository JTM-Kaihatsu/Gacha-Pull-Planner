const CHAR_DEFAULTS = { base_rate: 0.006, soft_pity_start: 73, hard_pity: 90 }
const LC_DEFAULTS   = { base_rate: 0.008, soft_pity_start: 65, hard_pity: 80 }

export { CHAR_DEFAULTS, LC_DEFAULTS }

function PityConfigRow({ label, config, onChange, rateMax, pityMax }) {
  return (
    <div>
      <div className="text-xs font-medium text-slate-400 mb-2">{label}</div>
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="block text-xs text-slate-500 mb-1">Base Rate (%)</label>
          <input
            type="number" min={0.1} max={rateMax} step={0.1}
            value={parseFloat((config.base_rate * 100).toFixed(2))}
            onChange={e => onChange({ ...config, base_rate: Math.max(0.001, +e.target.value / 100) })}
            onBlur={e => onChange({ ...config, base_rate: Math.max(0.001, Math.min(rateMax / 100, +e.target.value / 100)) })}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-sm focus:outline-none focus:border-violet-500"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">Soft Pity Start</label>
          <input
            type="number" min={1} max={pityMax - 1}
            value={config.soft_pity_start}
            onChange={e => onChange({ ...config, soft_pity_start: e.target.value === '' ? '' : +e.target.value })}
            onBlur={e => onChange({ ...config, soft_pity_start: Math.min(config.hard_pity - 1, Math.max(1, +e.target.value || 1)) })}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-sm focus:outline-none focus:border-violet-500"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">Hard Pity</label>
          <input
            type="number" min={2} max={pityMax}
            value={config.hard_pity}
            onChange={e => onChange({ ...config, hard_pity: e.target.value === '' ? '' : +e.target.value })}
            onBlur={e => onChange({ ...config, hard_pity: Math.min(pityMax, Math.max(config.soft_pity_start + 1, +e.target.value || pityMax)) })}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-white text-sm focus:outline-none focus:border-violet-500"
          />
        </div>
      </div>
    </div>
  )
}

export default function AdvancedSettings({ charConfig, lcConfig, onCharChange, onLcChange }) {
  return (
    <details className="group">
      <summary className="text-xs font-semibold text-slate-500 uppercase tracking-wider cursor-pointer select-none hover:text-slate-400 transition-colors">
        Advanced Settings
      </summary>
      <div className="mt-3 space-y-4 pl-1">
        <p className="text-xs text-slate-500">
          Override pity rate defaults. The soft pity increase per pull is derived automatically from these three values.
        </p>
        <PityConfigRow
          label="Character Banner"
          config={charConfig}
          onChange={onCharChange}
          rateMax={50}
          pityMax={180}
        />
        <PityConfigRow
          label="Light Cone Banner"
          config={lcConfig}
          onChange={onLcChange}
          rateMax={50}
          pityMax={180}
        />
      </div>
    </details>
  )
}
