import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term, Tooltip } from '../components/tooltip'
import { Empty, Meter, Panel, Stat } from '../components/ui'
import { bytes, count, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { GpuTelemetry, MemoryTelemetry } from '../lib/telemetry'

export function MemoryPanel({
  memory,
  gpu,
  replay,
}: {
  memory: MemoryTelemetry | undefined
  gpu: GpuTelemetry | undefined
  replay: boolean
}) {
  if (!memory) {
    return (
      <Panel title="VRAM" hint={replay ? 'replayMode' : 'vramBudget'}>
        <Empty>
          {replay ? 'live only — see the loaded log’s server_start record' : 'engine not attached'}
        </Empty>
      </Panel>
    )
  }

  // Arenas are reserved up front, so capacity - not used - is what the board actually holds.
  const lora = memory.lora_bank_bytes ?? 0
  const total =
    memory.weights.capacity_bytes +
    memory.sequence.capacity_bytes +
    memory.workspace.capacity_bytes +
    memory.cuda_graph_observed_bytes +
    lora +
    memory.available_after_startup_bytes
  const share = (value: number) =>
    `${bytes(value)} · ${percent(value / Math.max(1, total))} of board`
  const segments = [
    {
      label: 'weights',
      value: memory.weights.capacity_bytes,
      color: CHART.blue,
      hint: GLOSSARY.weightsArena.body,
      display: share(memory.weights.capacity_bytes),
    },
    {
      label: 'kv + state',
      value: memory.sequence.capacity_bytes,
      color: CHART.accent,
      hint: GLOSSARY.kvArena.body,
      display: share(memory.sequence.capacity_bytes),
    },
    {
      label: 'workspace',
      value: memory.workspace.capacity_bytes,
      color: CHART.violet,
      hint: GLOSSARY.workspaceArena.body,
      display: share(memory.workspace.capacity_bytes),
    },
    {
      label: 'graphs',
      value: memory.cuda_graph_observed_bytes,
      color: CHART.warning,
      hint: GLOSSARY.cudaGraphs.body,
      display: share(memory.cuda_graph_observed_bytes),
    },
    {
      label: 'lora',
      value: lora,
      color: CHART.danger,
      hint: GLOSSARY.adapterVram.body,
      display: share(lora),
    },
    {
      label: 'free',
      value: memory.available_after_startup_bytes,
      color: CHART.line,
      hint: 'Board memory left after every arena was reserved at startup.',
      display: share(memory.available_after_startup_bytes),
    },
  ]

  const kvUsed = memory.kv_payload_bytes
  const kvCapacity = Math.max(1, memory.sequence.capacity_bytes)

  const rows: Array<[string, number, string]> = [
    ['text kv', memory.text_kv_bytes, GLOSSARY.textKv.body],
    ['gdn state', memory.gdn_state_bytes, GLOSSARY.gdnState.body],
    ['mtp kv', memory.mtp_kv_bytes, GLOSSARY.mtpKv.body],
    ['dflash kv', memory.dflash_kv_bytes, 'KV cache for the text-only DFlash route.'],
    [
      'replay records',
      memory.replay_records_bytes,
      'Device-side records retained for state replay.',
    ],
    ['lora bank', lora, GLOSSARY.adapterBank.body],
  ]

  return (
    <Panel
      title="VRAM"
      hint="vramBudget"
      note={gpu?.available ? `${bytes(gpu.memory_used_bytes ?? 0)} resident on board` : undefined}
    >
      <div className="stat-row">
        <Stat value={bytes(memory.weights.capacity_bytes)} label="weights" hint="weightsArena" />
        <Stat
          value={bytes(memory.sequence.capacity_bytes)}
          label="kv arena"
          hint="kvArena"
          tone="accent"
        />
        <Stat
          value={bytes(memory.available_after_startup_bytes)}
          label="free"
          hint="vramBudget"
          tone={memory.available_after_startup_bytes < 512e6 ? 'warning' : 'neutral'}
        />
        <Stat value={count(memory.kv_capacity)} label="kv capacity" hint="kvArena" />
        <Stat value={count(memory.max_context)} label="max context" hint="pageGroups" />
      </div>

      <StackedBar segments={segments} height={10} />
      <Legend
        items={segments.map((segment) => ({
          label: segment.label,
          color: segment.color,
          hint: segment.hint,
        }))}
      />

      <div className="memory__kv">
        <div className="eyebrow">
          <Term k="kvPayload">kv payload</Term> {bytes(kvUsed)} of {bytes(kvCapacity)} arena (
          {percent(kvUsed / kvCapacity)})
        </div>
        <Meter fraction={kvUsed / kvCapacity} />
      </div>

      <table className="table">
        <tbody>
          {rows
            .filter(([, value]) => value > 0)
            .map(([label, value, hint]) => (
              <tr key={label}>
                <td>
                  <Tooltip title={label} body={hint} className="tip--term">
                    {label}
                  </Tooltip>
                </td>
                <td className="numeric emphasis">{bytes(value)}</td>
              </tr>
            ))}
          <tr>
            <td>
              <Term k="pageGroups">page groups</Term>
            </td>
            <td className="numeric emphasis">
              {count(memory.kv_capacity_page_groups)} / {count(memory.kv_capacity_max_page_groups)}
            </td>
          </tr>
          <tr>
            <td>
              <Term k="cudaGraphs">cuda graphs</Term>
            </td>
            <td className="numeric emphasis">
              {bytes(memory.cuda_graph_observed_bytes)} of{' '}
              {bytes(memory.cuda_graph_allowance_bytes)}
            </td>
          </tr>
          <tr>
            <td>
              <Term k="workspaceArena">workspace peak</Term>
            </td>
            <td className="numeric emphasis">
              {bytes(memory.workspace.peak_used_bytes)} of {bytes(memory.workspace.capacity_bytes)}
            </td>
          </tr>
        </tbody>
      </table>
    </Panel>
  )
}
