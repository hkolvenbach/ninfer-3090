import { Legend, SampleChart, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Meter, Panel, Stat } from '../components/ui'
import { count, percent, seconds } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { ServerStartRecord, ThroughputRecord } from '../lib/records'
import { latest, samples } from '../lib/series'
import type { Telemetry } from '../lib/telemetry'

const WORKER_UNITS = [
  { key: 'decode', label: 'decode', color: CHART.accent, hint: GLOSSARY.workerDecode.body },
  { key: 'prefill', label: 'prefill', color: CHART.violet, hint: GLOSSARY.workerPrefill.body },
  { key: 'admission', label: 'admission', color: CHART.blue, hint: GLOSSARY.workerAdmission.body },
  { key: 'publish', label: 'publish', color: CHART.warning, hint: GLOSSARY.workerPublish.body },
  { key: 'upkeep', label: 'upkeep', color: CHART.dim, hint: GLOSSARY.workerUpkeep.body },
] as const

export function SchedulerPanel({
  telemetry,
  records,
  engine,
}: {
  telemetry: Telemetry | null
  records: ThroughputRecord[]
  /** Configuration from the log's own server_start, used when no live telemetry is attached. */
  engine: ServerStartRecord['engine'] | undefined
}) {
  const scheduler = telemetry?.scheduler
  const sample = latest(records)
  const running = scheduler?.running ?? sample?.scheduler.running ?? 0
  const waiting = scheduler?.waiting ?? sample?.scheduler.waiting ?? 0
  const lanes = scheduler?.max_concurrency ?? engine?.max_concurrency ?? 1
  const pending = scheduler?.max_pending_requests ?? engine?.max_pending_requests ?? 0

  // Both the live snapshot and a throughput record carry the cumulative queue totals, so the mean
  // queue delay survives replay even though the live worker split does not.
  const queueTotals = scheduler ?? sample?.scheduler
  const meanQueue =
    queueTotals && queueTotals.admitted_requests > 0
      ? queueTotals.queue_seconds_total / queueTotals.admitted_requests
      : 0

  const worker = scheduler?.worker_seconds
  const admission = scheduler?.worker_admission
  const workerTotal = worker ? WORKER_UNITS.reduce((total, unit) => total + worker[unit.key], 0) : 0

  const rejected =
    (queueTotals?.rejected_overloaded ?? 0) + (queueTotals?.rejected_queue_timeout ?? 0)

  return (
    <Panel
      title="Scheduler"
      note={`${lanes} lane${lanes === 1 ? '' : 's'} · ${pending} pending slots`}
    >
      <div className="stat-row">
        <Stat
          value={`${running}/${lanes}`}
          label="running"
          hint="lanes"
          tone={running >= lanes ? 'warning' : 'accent'}
        />
        <Stat
          value={`${waiting}`}
          label="waiting"
          hint="queued"
          tone={waiting >= pending ? 'danger' : waiting > 0 ? 'warning' : 'neutral'}
        />
        <Stat
          value={seconds(meanQueue)}
          label="mean queue"
          hint="meanQueue"
          tone={meanQueue > 1 ? 'warning' : 'neutral'}
        />
        <Stat
          value={count(sample?.decode_batch.average_size ?? 0)}
          label="decode batch"
          hint="decodeBatch"
        />
        <Stat
          value={count(rejected)}
          label="rejected"
          hint="rejected"
          tone={rejected > 0 ? 'danger' : 'neutral'}
        />
      </div>

      <div className="scheduler__occupancy">
        <div className="eyebrow">
          <Term k="laneOccupancy">lane occupancy</Term>
        </div>
        <Meter fraction={lanes === 0 ? 0 : running / lanes} />
        <div className="eyebrow">
          <Term k="ingressQueue">ingress queue</Term>
        </div>
        <Meter
          fraction={pending === 0 ? 0 : waiting / pending}
          color={waiting >= pending ? 'var(--danger)' : 'var(--warning)'}
        />
      </div>

      {records.length === 0 ? (
        <Empty>no scheduler samples yet</Empty>
      ) : (
        <SampleChart
          label="Running requests and queue depth at each report"
          series={[
            {
              name: 'waiting',
              samples: samples(records, (record) => record.scheduler.waiting),
              color: CHART.warning,
            },
            {
              name: 'running',
              samples: samples(records, (record) => record.scheduler.running),
              color: CHART.blue,
            },
          ]}
          ceiling={lanes}
          legend={
            <Legend
              items={[
                { label: 'running', color: CHART.blue, hint: GLOSSARY.lanes.body },
                { label: 'waiting', color: CHART.warning, hint: GLOSSARY.queued.body },
              ]}
            />
          }
          caption={
            <>
              snapshots, not interval means <Info k="snapshotSeries" />
            </>
          }
        />
      )}

      {worker && workerTotal > 0 ? (
        <div className="scheduler__worker">
          <div className="eyebrow">
            <Term k="workerSplit">execution thread wall clock</Term>
          </div>
          <StackedBar
            segments={WORKER_UNITS.map((unit) => ({
              label: unit.label,
              value: worker[unit.key],
              color: unit.color,
              hint: unit.hint,
              display: `${seconds(worker[unit.key])} · ${percent(worker[unit.key] / workerTotal)} of thread time`,
            }))}
          />
          <Legend
            items={WORKER_UNITS.map((unit) => ({
              label: unit.label,
              color: unit.color,
              hint: unit.hint,
            }))}
          />
          <p className="panel__footnote">
            One mutex serializes these units, so a second spent in any of them is a second no other
            resident lane advances. Prefill share rising against decode is what starves a decoding
            request.{' '}
            {WORKER_UNITS.map(
              (unit) => `${unit.label} ${percent(worker[unit.key] / workerTotal)}`,
            ).join(' · ')}
          </p>
          {admission && admission.calls > 0 ? (
            <p className="panel__footnote">
              <Term k="admissionSplit">Admission</Term> over {count(admission.calls)} calls: plan{' '}
              {seconds(admission.plan_seconds)} · restore {seconds(admission.restore_seconds)} ·
              commit {seconds(admission.commit_seconds)} — most calls admit nothing, so this is{' '}
              {seconds(worker.admission / admission.calls)} per call.
            </p>
          ) : null}
        </div>
      ) : null}
    </Panel>
  )
}
