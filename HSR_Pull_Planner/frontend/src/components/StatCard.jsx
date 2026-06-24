export default function StatCard({ label, value, highlight = false }) {
  return (
    <div className={`rounded-xl p-4 border ${
      highlight
        ? 'bg-violet-900/30 border-violet-700'
        : 'bg-slate-800/60 border-slate-700'
    }`}>
      <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-2xl font-semibold ${highlight ? 'text-violet-300' : 'text-white'}`}>
        {value ?? '—'}
      </div>
    </div>
  )
}
