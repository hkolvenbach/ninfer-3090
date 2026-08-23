import { Term, Tooltip } from '../components/tooltip'
import { Empty, Meter, Panel, StatusDot } from '../components/ui'
import { count, percent } from '../lib/format'
import type { SlotTelemetry } from '../lib/telemetry'

const STATE_HINT = {
  busy: 'Executing a request right now.',
  retained: 'Idle but still holding a resident session in VRAM, reusable with no import at all.',
  idle: 'Empty and immediately available to the next admitted request.',
} as const

export function SlotsPanel({
  slots,
  maxContext,
  replay,
}: {
  slots: SlotTelemetry[] | undefined
  maxContext: number
  replay: boolean
}) {
  if (!slots || slots.length === 0) {
    return (
      <Panel title="Slots" hint={replay ? 'replayMode' : 'lanes'}>
        <Empty>
          {replay ? 'live only — lane occupancy is not recorded' : 'engine not attached'}
        </Empty>
      </Panel>
    )
  }

  return (
    <Panel
      title="Slots"
      hint="lanes"
      note={`${slots.filter((slot) => slot.processing).length} busy of ${slots.length}`}
    >
      <table className="table">
        <thead>
          <tr>
            <th />
            <th>slot</th>
            <th className="numeric">prompt</th>
            <th className="numeric">
              <Term k="slotReused">reused</Term>
            </th>
            <th className="numeric">
              <Tooltip
                title="Context fill"
                body="Resident prompt tokens against the model's maximum context."
                className="tip--term"
              >
                ctx
              </Tooltip>
            </th>
            <th>
              <Term k="sessionDigest">session</Term>
            </th>
            <th className="numeric">
              <Term k="checkpoints">ckpt</Term>
            </th>
          </tr>
        </thead>
        <tbody>
          {slots.map((slot, index) => {
            const fill = maxContext === 0 ? 0 : slot.prompt_tokens / maxContext
            const state = slot.processing ? 'busy' : slot.retained ? 'retained' : 'idle'
            return (
              <tr key={index}>
                <td>
                  <StatusDot
                    tone={slot.processing ? 'accent' : slot.retained ? 'warning' : 'neutral'}
                  />
                </td>
                <td className="emphasis">
                  {index}
                  <Tooltip
                    title={state}
                    body={STATE_HINT[state]}
                    className="slots__state tip--term"
                  >
                    {state}
                  </Tooltip>
                </td>
                <td className="numeric emphasis">{count(slot.prompt_tokens)}</td>
                <td className="numeric">
                  {slot.prompt_tokens === 0
                    ? '—'
                    : percent(slot.cached_tokens / slot.prompt_tokens)}
                </td>
                <td className="numeric" style={{ width: 72 }}>
                  <Meter fraction={fill} color={fill > 0.9 ? 'var(--warning)' : 'var(--blue)'} />
                </td>
                <td className="slots__digest">{slot.session_digest || '—'}</td>
                <td className="numeric">{slot.checkpoints || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="panel__footnote">
        A <Term k="retainedLane">retained lane</Term> holds a resident session in VRAM (L1) that a
        matching prompt can reuse without any import. <Term k="slotReused">reused</Term> is the
        share of the prompt served from resident prefix.
      </p>
    </Panel>
  )
}
