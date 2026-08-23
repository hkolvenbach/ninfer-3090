import { Stat, type Tone } from '../components/ui'
import type { RequestSummary } from '../lib/derive'
import { count, percent, rate, seconds } from '../lib/format'
import type { ServerStartRecord, ThroughputRecord } from '../lib/records'
import { latest } from '../lib/series'
import type { Telemetry } from '../lib/telemetry'

/**
 * The glance row. Every reading here is one a maintainer would otherwise reconstruct by hand:
 * current rates, whether requests are queueing, whether the cache is doing its job, and whether
 * the board is the limit.
 */
export function Headline({
  telemetry,
  records,
  summary,
  engine,
}: {
  telemetry: Telemetry | null
  records: ThroughputRecord[]
  summary: RequestSummary
  engine: ServerStartRecord['engine'] | undefined
}) {
  const sample = latest(records)
  const scheduler = telemetry?.scheduler
  const gpu = telemetry?.gpu

  // A replayed log has no live snapshot, but its records and server_start carry the same
  // scheduler configuration and cumulative queue totals.
  const running = scheduler?.running ?? sample?.scheduler.running ?? 0
  const lanes = scheduler?.max_concurrency ?? engine?.max_concurrency ?? 0
  const waiting = scheduler?.waiting ?? sample?.scheduler.waiting ?? 0
  const pending = scheduler?.max_pending_requests ?? engine?.max_pending_requests ?? 0
  const waitingTone: Tone =
    waiting >= pending && pending > 0 ? 'danger' : waiting > 0 ? 'warning' : 'neutral'

  const queueTotals = scheduler ?? sample?.scheduler
  const meanQueue =
    queueTotals && queueTotals.admitted_requests > 0
      ? queueTotals.queue_seconds_total / queueTotals.admitted_requests
      : 0

  // Board readings are live-only. Reporting 0 for an unavailable sensor would be a measurement
  // that was never taken, so they read as absent instead.
  const board = gpu?.available === true

  return (
    <div className="headline">
      <Stat
        value={rate(sample?.throughput_tokens_per_second.decode ?? 0)}
        unit="tok/s"
        label="decode · all lanes"
        hint="decodeRate"
        tone="accent"
      />
      <Stat
        value={rate(sample?.throughput_tokens_per_second.prefill ?? 0)}
        unit="tok/s"
        label="prefill"
        hint="prefillRate"
      />
      <Stat
        value={`${running}/${lanes}`}
        label="lanes"
        hint="lanes"
        tone={lanes > 0 && running >= lanes ? 'warning' : 'neutral'}
      />
      <Stat value={count(waiting)} label="queued" hint="queued" tone={waitingTone} />
      <Stat
        value={seconds(meanQueue)}
        label="mean queue"
        hint="meanQueue"
        tone={meanQueue > 1 ? 'warning' : 'neutral'}
      />
      <Stat value={seconds(summary.ttft.p50)} label="ttft p50" hint="ttft" />
      <Stat
        value={percent(summary.prefillAvoided)}
        label="prefill avoided"
        hint="prefillAvoided"
        tone={summary.prefillAvoided > 0.5 ? 'accent' : 'neutral'}
      />
      <Stat
        value={percent(summary.speculative.acceptRate)}
        label="mtp accept"
        hint="mtpAccept"
        tone={summary.speculative.acceptRate > 0.5 ? 'accent' : 'neutral'}
      />
      <Stat
        value={board ? `${gpu.utilization_gpu_percent ?? 0}` : '—'}
        unit={board ? '%' : undefined}
        label="gpu"
        hint="gpuUtil"
        tone={board && (gpu.utilization_gpu_percent ?? 0) > 90 ? 'accent' : 'neutral'}
      />
      <Stat
        value={board ? `${gpu.temperature_c ?? 0}` : '—'}
        unit={board ? '°C' : undefined}
        label="temp"
        hint="gpuTemp"
        tone={
          !board
            ? 'neutral'
            : (gpu.temperature_c ?? 0) >= 83
              ? 'danger'
              : (gpu.temperature_c ?? 0) >= 75
                ? 'warning'
                : 'neutral'
        }
      />
    </div>
  )
}
