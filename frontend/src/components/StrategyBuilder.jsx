export function buildStrategy(desiredChars, desiredWeapons, weaponAfter) {
  const strategy = []

  // Weapon only
  if (desiredChars === 0) {
    strategy.push({ banner: 'weapon', copies: desiredWeapons })
    return strategy
  }

  // Char only
  if (desiredWeapons === 0) {
    strategy.push({ banner: 'char', copies: desiredChars })
    return strategy
  }

  // Single char copy — no ordering question, Weapon always follows
  if (desiredChars === 1) {
    strategy.push({ banner: 'char', copies: 1 })
    strategy.push({ banner: 'weapon', copies: desiredWeapons })
    return strategy
  }

  // Multiple chars + weapons — split around weaponAfter insertion point
  const charsBefore = weaponAfter
  const charsAfter  = desiredChars - charsBefore

  if (charsBefore > 0) strategy.push({ banner: 'char', copies: charsBefore })
  strategy.push({ banner: 'weapon', copies: 1 })
  if (charsAfter > 0)  strategy.push({ banner: 'char', copies: charsAfter })
  if (desiredWeapons > 1)  strategy.push({ banner: 'weapon', copies: desiredWeapons - 1 })

  return strategy
}

export function buildOrderingOptions(desiredChars) {
  const options = Array.from({ length: desiredChars - 1 }, (_, i) => ({
    value: i + 1,
    label: `After C${i}, pull weapon before C${i + 1}`,
  }))
  options.push({
    value: desiredChars,
    label: `After C${desiredChars - 1}, pull all characters first`,
  })
  return options
}

export default function StrategyBuilder({ desiredChars, desiredWeapons, weaponAfter, onChange, validationError }) {
  const showOrdering = desiredChars > 1 && desiredWeapons >= 1
  const orderingOptions = buildOrderingOptions(desiredChars)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-slate-400 mb-1">Character Copies</label>
          <select
            value={desiredChars}
            onChange={e => onChange('desiredChars', +e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-violet-500"
          >
            <option value={0}>None</option>
            {[1,2,3,4,5,6].map(n => (
              <option key={n} value={n}>C{n-1} ({n} {n === 1 ? 'copy' : 'copies'})</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm text-slate-400 mb-1">Weapon Copies</label>
          <select
            value={desiredWeapons}
            onChange={e => onChange('desiredWeapons', +e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-violet-500"
          >
            <option value={0}>None</option>
            {[1,2,3,4,5].map(n => (
              <option key={n} value={n}>W{n} ({n} {n === 1 ? 'copy' : 'copies'})</option>
            ))}
          </select>
        </div>
      </div>

      {validationError && (
        <p className="text-red-400 text-sm">{validationError}</p>
      )}

      {showOrdering && (
        <div>
          <label className="block text-sm text-slate-400 mb-2">
            When do you want to pull your first Weapon?
          </label>
          <div className="space-y-2">
            {orderingOptions.map(opt => (
              <label key={opt.value} className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="radio"
                  name="weaponAfter"
                  value={opt.value}
                  checked={weaponAfter === opt.value}
                  onChange={() => onChange('weaponAfter', opt.value)}
                  className="accent-violet-500"
                />
                <span className="text-sm text-slate-300 group-hover:text-white transition-colors">
                  {opt.label}
                </span>
              </label>
            ))}
          </div>
        </div>
      )}

      {(desiredChars > 0 || desiredWeapons > 0) && (
        <StrategyPreview
          desiredChars={desiredChars}
          desiredWeapons={desiredWeapons}
          weaponAfter={weaponAfter}
          showOrdering={showOrdering}
        />
      )}
    </div>
  )
}

export function StrategyPreview({ desiredChars, desiredWeapons, weaponAfter, showOrdering }) {
  const strategy = buildStrategy(desiredChars, desiredWeapons, showOrdering ? weaponAfter : desiredChars)
  const labels = strategy.map(p => p.banner === 'char' ? `${p.copies} Char` : `${p.copies} Weapon`)

  return (
    <div className="flex items-center flex-wrap gap-2 pt-1">
      <span className="text-xs text-slate-500 uppercase tracking-wider">Pull order:</span>
      {labels.map((label, i) => (
        <span key={i} className="flex items-center gap-2">
          <span className={`text-xs font-medium px-2 py-1 rounded-md ${
            label.includes('Char')
              ? 'bg-violet-900/50 text-violet-300 border border-violet-700'
              : 'bg-amber-900/50 text-amber-300 border border-amber-700'
          }`}>
            {label}
          </span>
          {i < labels.length - 1 && <span className="text-slate-600">→</span>}
        </span>
      ))}
    </div>
  )
}
