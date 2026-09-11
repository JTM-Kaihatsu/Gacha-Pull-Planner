// Renders the advisor's deterministically-built "parsed situation" as a
// sequence of colored pills, one line per pull-run/event, built entirely
// from session_state.py's structured output (never asserted by the model).
const PILL_STYLES = {
  default: 'bg-slate-700/50 border-slate-600 text-slate-300',
  green: 'bg-green-900/40 border-green-700/60 text-green-300',
  red: 'bg-red-900/40 border-red-700/60 text-red-300',
  cyan: 'bg-cyan-900/30 border-cyan-700/50 text-cyan-300',
  magenta: 'bg-fuchsia-900/30 border-fuchsia-700/50 text-fuchsia-300',
  light_green: 'bg-emerald-900/30 border-emerald-700/50 text-emerald-300',
}

// A native `title` attribute is unreliable here: it has a ~1s dwell delay
// before it appears, is unstyled (breaks out of the dark theme), and is
// rendered by the OS outside the page, not something a hover state can be
// relied on to actually surface. This is a real, instant, styled tooltip
// instead, shown via a CSS group-hover on a wrapping span.
function Pill({ pill }) {
  // A "group" pill wraps its own sub-pills in literal parentheses with a
  // plain light border and no background of its own, so a two-part figure
  // like "(ending pity - starting pity)" reads as one bracketed unit.
  if (pill.kind === 'group') {
    // A red group flags a pity conflict: the reported figure can't be
    // reconciled with the banner's starting pity, distinct from the plain
    // grey border used for an ordinary, non-flagged breakdown.
    const groupBorder = pill.color === 'red' ? 'border-red-700/60' : 'border-slate-600'
    const parenColor = pill.color === 'red' ? 'text-red-500' : 'text-slate-500'
    return (
      <span className={`inline-flex items-center gap-1 rounded border ${groupBorder} px-1.5 py-1`}>
        <span className={`text-xs font-mono ${parenColor}`}>(</span>
        {pill.pills.map((p, i) => (
          <Pill key={i} pill={p} />
        ))}
        <span className={`text-xs font-mono ${parenColor}`}>)</span>
      </span>
    )
  }

  const style = PILL_STYLES[pill.color] || PILL_STYLES.default
  const isOperator = pill.kind === 'operator'
  return (
    <span className="relative inline-flex group">
      <span
        className={`inline-flex items-center text-xs font-mono rounded border leading-none ${style} ${
          isOperator ? 'px-1.5 py-1' : 'px-2 py-1'
        }`}
      >
        {pill.value}
      </span>
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 w-max max-w-[220px] -translate-x-1/2 whitespace-normal rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-[11px] leading-snug text-slate-200 opacity-0 shadow-lg transition-opacity duration-100 group-hover:opacity-100"
      >
        {pill.tooltip}
      </span>
    </span>
  )
}

function Line({ line }) {
  return (
    <div className="mb-2.5 last:mb-0">
      <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1">{line.label}</div>
      {/* pl-3 gives wrapped continuation pills the same hanging indent as
          the row's own start, so a run that overflows onto a second visual
          line still reads as one operation. */}
      <div className="flex flex-wrap items-center gap-1.5 pl-3">
        {line.pills.map((pill, i) => (
          <Pill key={i} pill={pill} />
        ))}
      </div>
    </div>
  )
}

// A branching question ('if I win... if I lose...') reconciles into two or
// more independent scenarios sharing the same starting point; each gets its
// own labeled, visually separate section rather than being interleaved into
// one flat pill list, so the reader can tell at a glance which pills belong
// to which branch.
function Group({ group }) {
  return (
    <div className="mb-4 last:mb-0 rounded-lg border border-slate-700/70 bg-slate-950/30 px-3 py-3">
      <div className="text-xs font-medium text-violet-300 mb-2.5">{group.label}</div>
      {group.lines.map((line, i) => (
        <Line key={i} line={line} />
      ))}
    </div>
  )
}

// The exact form inputs that produced the baseline simulation, shown above
// the parsed/reconciled content so a reader can see the starting point the
// rest of the panel reasons from without scrolling back to the form. The
// pill text here is deliberately spelled out in full ("180 starting pulls",
// "20 pity", "GUARANTEE: FALSE") rather than bare numbers: this panel
// doubles as the reader's legend for the terser pills further down.
function StartingSituation({ baseline }) {
  const f = baseline.form
  const chars = baseline.desiredChars ?? 0
  const weapons = baseline.desiredWeapons ?? 0
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`

  const bannerRow = (name, pity, guaranteed) => [
    { kind: 'banner', value: name, color: 'default',
      tooltip: `The pity and guarantee on this row are for the ${name.toLowerCase()} banner` },
    { kind: 'number', value: `${pity} pity`, color: 'cyan',
      tooltip: `Pity already built up on the ${name.toLowerCase()} banner at the start` },
    { kind: 'flag', value: `GUARANTEE: ${guaranteed ? 'TRUE' : 'FALSE'}`,
      color: guaranteed ? 'green' : 'red',
      tooltip: `Whether the ${name.toLowerCase()} banner starts with a guaranteed next 5-star` },
  ]

  const rows = [
    [
      { kind: 'number', value: `${f.total_pulls} starting pulls`, color: 'cyan',
        tooltip: 'Total pulls available at the start of this simulation' },
      { kind: 'number', value: `${plural(chars, 'character')} desired`, color: 'cyan',
        tooltip: 'Copies of the character this simulation was aiming for' },
      { kind: 'number', value: `${plural(weapons, 'weapon')} desired`, color: 'cyan',
        tooltip: 'Copies of the weapon this simulation was aiming for' },
    ],
    bannerRow('CHARACTER', f.start_char_pity, f.start_char_guarantee),
    bannerRow('WEAPON', f.start_weapon_pity, f.start_weapon_guarantee),
  ]

  return (
    <div className="bg-slate-900/40 border border-slate-700 rounded-lg px-3 py-3 mb-3">
      <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-2">Starting Situation</div>
      <div className="space-y-1.5">
        {rows.map((pills, ri) => (
          <div key={ri} className="flex flex-wrap items-center gap-1.5">
            {pills.map((pill, i) => (
              <Pill key={i} pill={pill} />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

// A single {"banner", "attempt_number"} pity conflict, matching exactly one
// clarifying question. Keyed this way (not by array index) so an answer
// survives the conflicts array being rebuilt by a fresh re-extraction.
export function conflictKey(conflict) {
  return `${conflict.banner}:${conflict.attempt_number}`
}

// One "did you mean total including existing pity" block: the header
// (styled like Starting Situation / Parsed Situation above it), the
// red-flagged pills, the clarifying question, and a textarea for the
// user's answer. Purely controlled: FollowUpAdvisor owns the answer text
// and the eventual submission, this just renders one conflict.
function ConflictBlock({ conflict, answer, onAnswerChange }) {
  return (
    <div className="mb-4 last:mb-0">
      <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-1.5">{conflict.header}</div>
      <div className="flex flex-wrap items-center gap-1.5 pl-3 mb-2">
        {conflict.pills.map((pill, i) => (
          <Pill key={i} pill={pill} />
        ))}
      </div>
      <p className="text-xs text-red-300 pl-3 mb-2">{conflict.question}</p>
      <textarea
        value={answer}
        onChange={e => onAnswerChange(e.target.value)}
        placeholder="Type your answer here…"
        rows={2}
        className="w-full bg-slate-900/60 border border-red-800/50 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-red-500 resize-none"
      />
    </div>
  )
}

export default function ParsedSituation({ breakdown, baseline, conflictAnswers, onConflictAnswerChange }) {
  const startingSituation = baseline && <StartingSituation baseline={baseline} />

  if (!breakdown) return startingSituation || null

  if (breakdown.status === 'error') {
    return (
      <>
        {startingSituation}
        <div className="bg-red-950/30 border border-red-800/50 rounded-lg px-3 py-2 mb-3">
          <div className="text-[11px] text-red-400 uppercase tracking-wider mb-1">Parsed Situation</div>
          <p className="text-xs text-red-300">{breakdown.message}</p>
        </div>
      </>
    )
  }

  if (breakdown.status === 'conflict') {
    return (
      <>
        {startingSituation}
        {/* Whatever preceded the first conflict is already fully resolved,
            not in question, so it renders exactly like a normal Parsed
            Situation panel; empty when the very first event conflicts. */}
        {breakdown.lines.length > 0 && (
          <div className="bg-slate-900/40 border border-slate-700 rounded-lg px-3 py-3 mb-3">
            <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-2">Parsed Situation</div>
            {breakdown.lines.map((line, i) => <Line key={i} line={line} />)}
          </div>
        )}
        <div className="bg-red-950/20 border border-red-800/40 rounded-lg px-3 py-3 mb-3">
          {breakdown.conflicts.map(conflict => {
            const key = conflictKey(conflict)
            return (
              <ConflictBlock
                key={key}
                conflict={conflict}
                answer={conflictAnswers?.[key] || ''}
                onAnswerChange={value => onConflictAnswerChange?.(conflict, value)}
              />
            )
          })}
        </div>
      </>
    )
  }

  return (
    <>
      {startingSituation}
      <div className="bg-slate-900/40 border border-slate-700 rounded-lg px-3 py-3 mb-3">
        <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-2">Parsed Situation</div>
        {breakdown.groups
          ? breakdown.groups.map((group, i) => <Group key={i} group={group} />)
          : breakdown.lines.map((line, i) => <Line key={i} line={line} />)}
      </div>
    </>
  )
}
