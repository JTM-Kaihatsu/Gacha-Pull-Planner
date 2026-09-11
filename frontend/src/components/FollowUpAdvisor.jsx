import { useEffect, useRef, useState } from 'react'
import { advise } from '../api'
import { buildScenarioPayload, suggestedQuestions } from '../lib/scenarios'
import ParsedSituation, { conflictKey } from './ParsedSituation'

// Suggested questions may carry a literal [PLACEHOLDER] the user is meant
// to replace with their own number, not an illustrative one: a concrete-
// but-fake number would look exactly like a real answer if sent unedited.
const PLACEHOLDER_PATTERN = /\[[^\]]+\]/

const MAX_QUESTION_LENGTH = 500

const STATUS_MESSAGE = {
  rate_limited: 'The advisor is temporarily unavailable (rate limited). Try again in a little while.',
  unavailable: "The advisor could not answer right now. Try again later.",
}

export default function FollowUpAdvisor({ baseline, confidence }) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [statusMessage, setStatusMessage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  // Per-conflict answer text, keyed by conflictKey(conflict); cleared on a
  // fresh Ask or once a clarification submission resolves.
  const [conflictAnswers, setConflictAnswers] = useState({})
  // How the AI resolved a prior conflict (label + text), shown between the
  // textarea and the Ask button. Kept separate from `breakdown` because it
  // must disappear as soon as the user starts typing a new question, even
  // before they click Ask again, while `breakdown` itself only resets then.
  const [annotations, setAnnotations] = useState([])

  const suggestions = suggestedQuestions(confidence, baseline)
  const trimmed = question.trim()
  const canAsk = trimmed.length > 0 && !loading
  const textareaRef = useRef(null)
  const pendingSelectionRef = useRef(null)

  function handleSuggestionClick(s) {
    setQuestion(s)
    const match = s.match(PLACEHOLDER_PATTERN)
    pendingSelectionRef.current = match ? [match.index, match.index + match[0].length] : null
  }

  // Applying the selection here (after the DOM has actually picked up the
  // new value) is reliable; guessing a frame via requestAnimationFrame in
  // the click handler above raced React's own re-render and lost.
  useEffect(() => {
    const pending = pendingSelectionRef.current
    const el = textareaRef.current
    if (!pending || !el) return
    el.focus()
    el.setSelectionRange(pending[0], pending[1])
    pendingSelectionRef.current = null
  }, [question])

  // Clicking Ask again is a chance to modify the question rather than start
  // anew: it always resets the panel, including any pending conflict UI and
  // its answers, whatever the textarea currently holds.
  async function handleAsk() {
    if (!canAsk) return
    setLoading(true)
    setError(null)
    setAnswer(null)
    setBreakdown(null)
    setStatusMessage(null)
    setConflictAnswers({})
    setAnnotations([])
    try {
      const payload = { ...buildScenarioPayload(baseline, {}), question: trimmed }
      const data = await advise(payload)
      applyAdviseResponse(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // A pity conflict answers with an empty answer_text ("" is falsy), the
  // conflict blocks themselves are the response, so a plain truthiness
  // check on data.answer alone would wrongly swallow that case.
  function applyAdviseResponse(data) {
    if (data.status === 'ok' && (data.answer || data.breakdown?.status === 'conflict')) {
      setAnswer(data.answer || null)
      setBreakdown(data.breakdown || null)
    } else {
      setStatusMessage(STATUS_MESSAGE[data.status] || STATUS_MESSAGE.unavailable)
    }
  }

  function handleConflictAnswerChange(conflict, value) {
    setConflictAnswers(prev => ({ ...prev, [conflictKey(conflict)]: value }))
  }

  // Every filled-in answer is sent together, the same retry the backend
  // performs in one shot; the resolved (or, if still ambiguous, freshly
  // re-flagged) response replaces the panel in place, an auto-resubmit
  // rather than a second manual Ask.
  async function handleSubmitClarifications() {
    if (!breakdown || breakdown.status !== 'conflict' || loading) return
    const clarifications = breakdown.conflicts
      .map(c => ({ banner: c.banner, attempt_number: c.attempt_number,
                   answer: (conflictAnswers[conflictKey(c)] || '').trim() }))
      .filter(c => c.answer.length > 0)
    if (clarifications.length === 0) return

    setLoading(true)
    setError(null)
    setStatusMessage(null)
    try {
      const payload = { ...buildScenarioPayload(baseline, {}), question: trimmed, clarifications }
      const data = await advise(payload)
      applyAdviseResponse(data)
      setConflictAnswers({})
      setAnnotations(data.breakdown?.annotations || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <details className="bg-slate-800/40 border border-slate-700 rounded-xl p-4">
      <summary className="text-sm font-medium text-slate-400 select-none cursor-pointer">Ask a Follow-up (AI)</summary>
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
                onClick={() => handleSuggestionClick(s)}
                className="text-xs text-left px-3 py-1.5 rounded-lg border bg-slate-800 border-slate-700 text-slate-300 hover:border-violet-500 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <textarea
          ref={textareaRef}
          value={question}
          onChange={e => {
            setQuestion(e.target.value.slice(0, MAX_QUESTION_LENGTH))
            // The labeled context line only stays visible until a new
            // question is typed in, even before Ask is clicked again.
            if (annotations.length > 0) setAnnotations([])
          }}
          placeholder="Ask your own, or click a suggestion above to start."
          rows={2}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-violet-500 resize-none"
        />

        {annotations.length > 0 && (
          <div className="space-y-1">
            {annotations.map((a, i) => (
              <p key={i} className="text-xs text-slate-400">
                <span className="font-bold text-orange-400">{a.label}</span>
                {a.text}
              </p>
            ))}
          </div>
        )}

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
        {(answer || breakdown?.status === 'conflict') && (
          <div className="bg-slate-900/60 border border-slate-700 rounded-lg p-4">
            <div className="text-xs text-violet-400 uppercase tracking-wider mb-2">Advisor</div>
            {/* Every simulation the advisor ran (the guaranteed pre-run and any
                further exploration) is already represented inside breakdown.lines
                as its own labeled pill line ("Simulated Result" / "Agent Run
                Cycle N"), so there is no separate receipt-chip list here. */}
            <ParsedSituation
              breakdown={breakdown}
              baseline={baseline}
              conflictAnswers={conflictAnswers}
              onConflictAnswerChange={handleConflictAnswerChange}
            />
            {breakdown?.status === 'conflict' ? (
              <button
                type="button"
                onClick={handleSubmitClarifications}
                disabled={loading || Object.values(conflictAnswers).every(a => !a.trim())}
                className="text-sm font-medium px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
              >
                {loading ? 'Thinking…' : 'Submit Clarification'}
              </button>
            ) : (
              <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{answer}</p>
            )}
          </div>
        )}
      </div>
    </details>
  )
}
