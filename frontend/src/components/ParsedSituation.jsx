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

export default function ParsedSituation({ breakdown }) {
  if (!breakdown) return null

  if (breakdown.status === 'error') {
    return (
      <div className="bg-red-950/30 border border-red-800/50 rounded-lg px-3 py-2 mb-3">
        <div className="text-[11px] text-red-400 uppercase tracking-wider mb-1">Parsed Situation</div>
        <p className="text-xs text-red-300">{breakdown.message}</p>
      </div>
    )
  }

  return (
    <div className="bg-slate-900/40 border border-slate-700 rounded-lg px-3 py-3 mb-3">
      <div className="text-[11px] text-slate-500 uppercase tracking-wider mb-2">Parsed Situation</div>
      {breakdown.groups
        ? breakdown.groups.map((group, i) => <Group key={i} group={group} />)
        : breakdown.lines.map((line, i) => <Line key={i} line={line} />)}
    </div>
  )
}
