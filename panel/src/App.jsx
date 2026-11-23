import { useEffect, useMemo, useRef, useState } from 'react'

const STATUS_LABEL = {
  idle: 'No task yet',
  running: 'Running',
  awaiting_approval: 'Waiting for your approval',
  completed: 'Completed',
  budget_exceeded: 'Stopped: budget',
  failed: 'Stopped',
}

export default function App() {
  const [meta, setMeta] = useState(null)
  const [goal, setGoal] = useState('')
  const [role, setRole] = useState('lead')
  const [runId, setRunId] = useState(null)
  const [events, setEvents] = useState([])
  const [status, setStatus] = useState('idle')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const sourceRef = useRef(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    fetch('/api/meta').then(r => r.json()).then(setMeta).catch(() => setError('Cannot reach the server. Is python server.py running?'))
  }, [])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [events])

  async function start(e) {
    e?.preventDefault()
    if (!goal.trim()) return
    sourceRef.current?.close()
    setEvents([]); setError(''); setStatus('running'); setBusy(true)
    const res = await fetch('/api/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal, role }),
    })
    if (!res.ok) { setError(`Could not start: ${res.status}`); setBusy(false); setStatus('idle'); return }
    const { run_id } = await res.json()
    setRunId(run_id)
    const es = new EventSource(`/api/runs/${run_id}/events`)
    sourceRef.current = es
    es.onmessage = msg => {
      const ev = JSON.parse(msg.data)
      setEvents(prev => [...prev, ev])
      if (ev.type === 'state') setStatus(ev.status)
      if (ev.type === 'approval') setStatus('running')
      if (ev.type === 'done') { setStatus(s => (ev.status === 'failed' ? 'failed' : s)); setBusy(false); es.close() }
    }
    es.onerror = () => { setBusy(false); es.close() }
  }

  async function decide(approved, reason) {
    const res = await fetch(`/api/runs/${runId}/decision`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved, reason }),
    })
    if (res.ok) setStatus('running')
  }

  const started = events.find(e => e.type === 'run_started')
  const stats = useMemo(() => computeStats(events, started), [events, started])

  return (
    <div className="app">
      <header>
        <div>
          <h1>Agent Control Panel</h1>
          <p className="sub">Banking operations agent · LangGraph + MCP · {meta ? `${meta.model}, prompt ${meta.prompt_version}` : 'loading'}</p>
        </div>
        <span className={`badge status-${status}`}>{STATUS_LABEL[status] ?? status}</span>
      </header>

      <main>
        <aside className="left">
          <form onSubmit={start} className="card">
            <h2>New task</h2>
            <label>Run as
              <select value={role} onChange={e => setRole(e.target.value)} disabled={busy}>
                {(meta?.roles ?? []).map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
              </select>
            </label>
            <label>Goal
              <textarea rows={5} value={goal} onChange={e => setGoal(e.target.value)} disabled={busy}
                placeholder="e.g. Check whether ACC-102 exceeds EUR 10,000 and open a ticket if it does." />
            </label>
            <button type="submit" disabled={busy || !goal.trim()}>{busy ? 'Running…' : 'Run task'}</button>
            {error && <p className="error">{error}</p>}
          </form>
          <div className="card">
            <h2>Try an example</h2>
            <ul className="examples">
              {(meta?.examples ?? []).map(ex => (
                <li key={ex.goal}>
                  <button type="button" disabled={busy} onClick={() => { setGoal(ex.goal); setRole(ex.role) }}>
                    <span className="role-tag">{ex.role === 'ops' ? 'Ops' : 'Lead'}</span>{ex.goal}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </aside>

        <section className="timeline">
          {events.length === 0 && (
            <div className="empty">
              <p>Pick a role and a goal, then press <b>Run task</b>.</p>
              <p>You will see each decision the model makes, every tool call and its result, and you approve or reject anything that changes data.</p>
            </div>
          )}
          {events.map((ev, i) => (
            <Event key={i} ev={ev} isLatestApproval={ev.type === 'approval_request' && status === 'awaiting_approval' &&
              !events.slice(i + 1).some(x => x.type === 'approval')} onDecide={decide} />
          ))}
          <div ref={bottomRef} />
        </section>

        <aside className="right">
          <div className="card">
            <h2>Budgets</h2>
            {started ? (
              <>
                <Meter label="Tool steps" value={stats.steps} max={started.budgets.max_steps} />
                <Meter label="Model calls" value={stats.llmCalls} max={started.budgets.max_llm_calls} />
                <Meter label="Cost (hosted equiv.)" value={stats.cost} max={started.budgets.max_cost_usd}
                  format={v => `$${v.toFixed(6)}`} />
                <dl className="facts">
                  <dt>Tokens</dt><dd>{stats.tokens.toLocaleString()}</dd>
                  <dt>Guardrail blocks</dt><dd>{stats.guards}</dd>
                  <dt>Elapsed</dt><dd>{stats.elapsed}s</dd>
                </dl>
              </>
            ) : <p className="muted">Shown once a task starts.</p>}
          </div>
          <div className="card">
            <h2>Tools for this role</h2>
            {started ? (
              <ul className="tools">
                {started.tools.map(t => (
                  <li key={t} className={stats.used.has(t) ? 'used' : ''}>
                    {t}{started.gated_tools.includes(t) && <span className="lock" title="Needs approval">approval</span>}
                  </li>
                ))}
              </ul>
            ) : <p className="muted">The MCP server only lists the tools the chosen role may use.</p>}
          </div>
        </aside>
      </main>
    </div>
  )
}

function computeStats(events, started) {
  const pricing = started?.pricing ?? { usd_per_million_input_tokens: 0, usd_per_million_output_tokens: 0 }
  let steps = 0, llmCalls = 0, inTok = 0, outTok = 0, guards = 0
  const used = new Set()
  for (const ev of events) {
    if (ev.type === 'tool_result') { steps++; used.add(ev.tool) }
    if (ev.type === 'tool_call' || ev.type === 'final') { llmCalls++; inTok += ev.prompt_tokens ?? 0; outTok += ev.completion_tokens ?? 0 }
    if (ev.type === 'guard') guards++
  }
  const cost = (inTok * pricing.usd_per_million_input_tokens + outTok * pricing.usd_per_million_output_tokens) / 1e6
  const first = events[0]?.t, last = events[events.length - 1]?.t
  return { steps, llmCalls, tokens: inTok + outTok, cost, guards, used, elapsed: first && last ? (last - first).toFixed(1) : '0.0' }
}

function Meter({ label, value, max, format = v => v }) {
  const pct = Math.min(100, max ? (value / max) * 100 : 0)
  return (
    <div className="meter">
      <div className="meter-label"><span>{label}</span><span>{format(value)} / {format(max)}</span></div>
      <div className="bar"><div className={pct >= 100 ? 'fill over' : 'fill'} style={{ width: `${pct}%` }} /></div>
    </div>
  )
}

function preview(result) {
  if (!result || typeof result !== 'object') return String(result ?? '')
  const text = Object.entries(result)
    .map(([k, v]) => `${k}: ${Array.isArray(v) ? `[${v.length}]` : typeof v === 'object' && v ? '{…}' : v}`)
    .join(' · ')
  return text.length > 150 ? text.slice(0, 150) + '…' : text
}

function Json({ value }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>
}

function Event({ ev, isLatestApproval, onDecide }) {
  const [reason, setReason] = useState('')
  switch (ev.type) {
    case 'goal':
      return <div className="ev goal"><span className="kind">Task</span><p>{ev.text}</p></div>
    case 'thought':
      return <div className="ev thought"><span className="kind">Model</span><p>{ev.text}</p></div>
    case 'tool_call':
      return (
        <div className="ev call">
          <span className="kind">Step {ev.step} · decides</span>
          <p>Call <code>{ev.tool}</code></p>
          <Json value={ev.args} />
        </div>
      )
    case 'tool_result':
      return (
        <div className={`ev result ${ev.ok ? 'ok' : 'bad'}`}>
          <span className="kind">{ev.ok ? 'Result' : `Tool error · ${ev.error_type}`}</span>
          <details open={!ev.ok}>
            <summary><code>{ev.tool}</code> <span className="preview">{preview(ev.result)}</span></summary>
            <Json value={ev.result} />
          </details>
        </div>
      )
    case 'guard':
      return <div className="ev guard"><span className="kind">Guardrail · {ev.error_type}</span><p>Blocked <code>{ev.tool}</code>. {ev.text}</p></div>
    case 'approval_request':
      return (
        <div className={`ev approval ${isLatestApproval ? 'live' : ''}`}>
          <span className="kind">Needs approval</span>
          <p>The agent wants to run <code>{ev.tool}</code> with the arguments above. It changes data, so it waits for you.</p>
          {isLatestApproval && (
            <div className="decide">
              <input placeholder="Reason (optional)" value={reason} onChange={e => setReason(e.target.value)} />
              <button className="approve" onClick={() => onDecide(true, reason)}>Approve</button>
              <button className="reject" onClick={() => onDecide(false, reason)}>Reject</button>
            </div>
          )}
        </div>
      )
    case 'approval':
      return (
        <div className={`ev ${ev.approved ? 'approved' : 'rejected'}`}>
          <span className="kind">{ev.approved ? 'Approved' : 'Rejected'}</span>
          <p>{ev.approved ? 'You approved' : 'You rejected'} <code>{ev.tool}</code>{ev.reason ? `: ${ev.reason}` : '.'}</p>
        </div>
      )
    case 'final':
      return <div className="ev final"><span className="kind">Answer</span><p>{ev.text}</p></div>
    case 'stopped':
      return <div className="ev stopped"><span className="kind">Stopped</span><p>{ev.reason}</p></div>
    case 'error':
      return <div className="ev stopped"><span className="kind">Error</span><p>{ev.text}</p><p className="muted">Is Ollama running with the model pulled?</p></div>
    default:
      return null
  }
}
