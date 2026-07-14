import { useState } from 'react'
import { advise } from '../api'
import { buildScenarioPayload, suggestedQuestions } from '../lib/scenarios'

const MAX_QUESTION_LENGTH = 500

const STATUS_MESSAGE = {
  rate_limited: 'The advisor is temporarily unavailable (rate limited). Try again in a little while.',
  unavailable: "The advisor could not answer right now. Try again later.",
}

export default function FollowUpAdvisor({ baseline, confidence }) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [runs, setRuns] = useState([])
  const [statusMessage, setStatusMessage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const suggestions = suggestedQuestions(confidence)
  const trimmed = question.trim()
  const canAsk = trimmed.length > 0 && !loading

  async function handleAsk() {
    if (!canAsk) return
    setLoading(true)
    setError(null)
    setAnswer(null)
    setRuns([])
    setStatusMessage(null)
    try {
      const payload = { ...buildScenarioPayload(baseline, {}), question: trimmed }
      const data = await advise(payload)
      if (data.status === 'ok' && data.answer) {
        setAnswer(data.answer)
        setRuns(data.runs || [])
      } else {
        setStatusMessage(STATUS_MESSAGE[data.status] || STATUS_MESSAGE.unavailable)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <details className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 cursor-pointer">
      <summary className="text-sm font-medium text-slate-400 select-none">Ask a Follow-up (AI)</summary>
      <div className="mt-4 space-y-3">
        <p className="text-xs text-slate-500">
          Ask an open-ended what-if the presets do not cover. The AI re-runs the
          simulation to answer, so this uses the OpenAI API and may take a moment.
        </p>

        <div>
          <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">Suggestions</div>
          <div className="flex flex-wrap gap-2">
            {suggestions.map((s, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setQuestion(s)}
                className="text-xs text-left px-3 py-1.5 rounded-lg border bg-slate-800 border-slate-700 text-slate-300 hover:border-violet-500 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <textarea
          value={question}
          onChange={e => setQuestion(e.target.value.slice(0, MAX_QUESTION_LENGTH))}
          placeholder="Ask your own, or click a suggestion above to start."
          rows={2}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-violet-500 resize-none"
        />

        <div className="flex items-center justify-between gap-3">
          <span className="text-xs text-slate-600">{question.length}/{MAX_QUESTION_LENGTH}</span>
          <button
            type="button"
            onClick={handleAsk}
            disabled={!canAsk}
            className="text-sm font-medium px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
          >
            {loading ? 'Thinking…' : 'Ask'}
          </button>
        </div>

        {error && <p className="text-red-400 text-sm">{error}</p>}
        {statusMessage && <p className="text-slate-400 text-sm italic">{statusMessage}</p>}
        {answer && (
          <div className="bg-slate-900/60 border border-slate-700 rounded-lg p-4">
            <div className="text-xs text-violet-400 uppercase tracking-wider mb-2">Advisor</div>
            {runs.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {runs.map((r, i) => (
                  <span
                    key={i}
                    className="text-xs font-mono bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-400"
                    title="A simulation the advisor ran to answer"
                  >
                    {r.total_pulls} pulls, {r.desired_characters}C/{r.desired_weapons}W {'→'} {r.success_rate}
                  </span>
                ))}
              </div>
            )}
            <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{answer}</p>
          </div>
        )}
      </div>
    </details>
  )
}
