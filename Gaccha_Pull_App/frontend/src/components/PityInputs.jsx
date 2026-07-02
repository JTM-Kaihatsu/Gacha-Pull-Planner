export default function PityInputs({ form, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <label className="block text-sm text-slate-400 mb-1">Char Pity <span className="text-slate-500">(0–90)</span></label>
        <input
          type="number" min={0} max={90}
          value={form.start_char_pity}
          onChange={e => onChange('start_char_pity', e.target.value === '' ? '' : +e.target.value)}
          onBlur={e => onChange('start_char_pity', Math.min(90, Math.max(0, +e.target.value || 0)))}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-violet-500"
        />
        <label className="flex items-center gap-2 mt-2 text-sm text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={form.start_char_guarantee}
            onChange={e => onChange('start_char_guarantee', e.target.checked)}
            className="accent-violet-500"
          />
          Guaranteed next 5★
        </label>
      </div>

      <div>
        <label className="block text-sm text-slate-400 mb-1">Weapon Pity <span className="text-slate-500">(0–80)</span></label>
        <input
          type="number" min={0} max={80}
          value={form.start_weapon_pity}
          onChange={e => onChange('start_weapon_pity', e.target.value === '' ? '' : +e.target.value)}
          onBlur={e => onChange('start_weapon_pity', Math.min(80, Math.max(0, +e.target.value || 0)))}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-violet-500"
        />
        <label className="flex items-center gap-2 mt-2 text-sm text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={form.start_weapon_guarantee}
            onChange={e => onChange('start_weapon_guarantee', e.target.checked)}
            className="accent-violet-500"
          />
          Guaranteed next 5★ Weapon
        </label>
      </div>
    </div>
  )
}
