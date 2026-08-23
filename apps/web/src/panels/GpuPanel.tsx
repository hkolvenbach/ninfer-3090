import { Legend, LineChart } from '../components/charts'
import { CHART } from '../components/echart'
import { Term } from '../components/tooltip'
import { Empty, Meter, Panel, Pill, Stat } from '../components/ui'
import type { GpuSample } from '../lib/engine-client'
import { bytes, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import { activeThrottles, type GpuTelemetry } from '../lib/telemetry'

export function GpuPanel({
  gpu,
  history,
  replay,
}: {
  gpu: GpuTelemetry | undefined
  history: GpuSample[]
  replay: boolean
}) {
  if (!gpu || !gpu.available) {
    return (
      <Panel title="GPU" hint={replay ? 'replayMode' : 'gpuUtil'} note="NVML">
        {/* Board telemetry is sampled live and is not part of the record schema, so a replayed
            file genuinely cannot supply it. */}
        <Empty>
          {replay
            ? 'live only — a request log carries no board telemetry'
            : gpu?.error
              ? `NVML unavailable: ${gpu.error}`
              : 'no board telemetry'}
        </Empty>
      </Panel>
    )
  }

  const throttles = activeThrottles(gpu)
  const temperature = gpu.temperature_c ?? 0
  const power = gpu.power_watts ?? 0
  const powerLimit = gpu.power_limit_watts ?? 0
  const memoryUsed = gpu.memory_used_bytes ?? 0
  const memoryTotal = gpu.memory_total_bytes ?? 1
  const clockFraction = (gpu.sm_clock_mhz ?? 0) / Math.max(1, gpu.sm_clock_max_mhz ?? 1)

  return (
    <Panel
      title="GPU"
      note={
        <>
          {gpu.name}
          {throttles.length > 0 ? (
            <>
              {' '}
              <Term k="gpuThrottle">
                <Pill tone="danger">{throttles.join(' · ')}</Pill>
              </Term>
            </>
          ) : null}
        </>
      }
    >
      <div className="stat-row">
        <Stat
          value={`${gpu.utilization_gpu_percent ?? 0}`}
          unit="%"
          label="sm util"
          hint="gpuUtil"
          tone="accent"
        />
        <Stat
          value={`${temperature}`}
          unit="°C"
          label="temp"
          hint="gpuTemp"
          tone={temperature >= 83 ? 'danger' : temperature >= 75 ? 'warning' : 'neutral'}
        />
        <Stat
          value={power.toFixed(0)}
          unit={`/ ${powerLimit.toFixed(0)} W`}
          label="power"
          hint="gpuPower"
          tone={powerLimit > 0 && power / powerLimit > 0.97 ? 'warning' : 'neutral'}
        />
        <Stat
          value={`${gpu.sm_clock_mhz ?? 0}`}
          unit="MHz"
          label="sm clock"
          hint="gpuClock"
          tone={clockFraction < 0.75 ? 'warning' : 'neutral'}
        />
        <Stat
          value={`${gpu.utilization_memory_percent ?? 0}`}
          unit="%"
          label="mem bw"
          hint="gpuMemBw"
        />
      </div>

      <div className="gpu__memory">
        <div className="eyebrow">
          <Term k="boardMemory">board memory</Term> {bytes(memoryUsed)} / {bytes(memoryTotal)} (
          {percent(memoryUsed / memoryTotal)})
        </div>
        <Meter
          fraction={memoryUsed / memoryTotal}
          color={memoryUsed / memoryTotal > 0.95 ? 'var(--warning)' : 'var(--blue)'}
        />
      </div>

      {history.length < 2 ? (
        <Empty>collecting board samples</Empty>
      ) : (
        <LineChart
          label="GPU utilization over the sampled window"
          name="sm utilization"
          unit="%"
          samples={history.map((sample) => ({ at: sample.at, value: sample.utilization }))}
          color={CHART.accent}
          ceiling={100}
          marks={history.map((sample) => sample.reasons)}
          legend={
            <Legend
              items={[
                { label: 'sm utilization %', color: CHART.accent, hint: GLOSSARY.gpuUtil.body },
                { label: 'throttled', color: CHART.danger, hint: GLOSSARY.gpuThrottle.body },
              ]}
            />
          }
          caption={`${history.length}s @ 1 Hz`}
        />
      )}
      <p className="panel__footnote">
        Sustained prefill on a 24 GB board can throttle and be misread as a kernel regression. Red
        rules mark samples with an active non-idle throttle reason.
      </p>
    </Panel>
  )
}
