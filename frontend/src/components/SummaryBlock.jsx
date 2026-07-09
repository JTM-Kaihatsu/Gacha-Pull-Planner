const CONFIDENCE_COLOR = {
  comfortable: 'text-emerald-400',
  likely: 'text-emerald-300',
  coin_flip: 'text-amber-300',
  stretch: 'text-orange-300',
  long_shot: 'text-red-400',
}

export default function SummaryBlock({ summary }) {
  if (!summary) return null
  const color = CONFIDENCE_COLOR[summary.confidence] || 'text-slate-200'

  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">
        The Read
      </h3>
      <p className={`text-lg font-semibold ${color} mb-2`}>{summary.headline}</p>
      <ul className="space-y-1.5">
        {summary.notes.map((note, i) => (
          <li key={i} className="text-sm text-slate-300 leading-relaxed">
            {note}
          </li>
        ))}
      </ul>
    </div>
  )
}
