export default function AnalysisBlock({ text }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-violet-400 uppercase tracking-wider mb-3">
        AI Analysis
      </h3>
      <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">{text}</p>
    </div>
  )
}
