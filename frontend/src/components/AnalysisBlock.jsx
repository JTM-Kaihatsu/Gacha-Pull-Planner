const UNAVAILABLE_MESSAGES = {
  rate_limited:
    'The AI verdict is temporarily unavailable — the service is rate-limited right now. Your simulation results above are complete; try the AI verdict again in a little while.',
  unavailable:
    "The AI verdict couldn't be generated right now. Your simulation results above are complete; try again later.",
}

export default function AnalysisBlock({ text, status = 'ok' }) {
  const unavailableMessage = UNAVAILABLE_MESSAGES[status]

  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5">
      <h3 className="text-sm font-semibold text-violet-400 uppercase tracking-wider mb-3">
        AI Analysis
      </h3>
      {unavailableMessage ? (
        <p className="text-slate-400 leading-relaxed italic">{unavailableMessage}</p>
      ) : (
        <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">{text}</p>
      )}
    </div>
  )
}
