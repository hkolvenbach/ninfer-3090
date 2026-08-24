import { BandChart, Legend } from '../components/charts'
import { CHART } from '../components/echart'
import { Info, Term } from '../components/tooltip'
import { Empty, Panel, Stat } from '../components/ui'
import { summarizeChurn } from '../lib/derive'
import { bytes, count, percent } from '../lib/format'
import { GLOSSARY } from '../lib/glossary'
import type { ThroughputRecord } from '../lib/records'
import { bands } from '../lib/series'
import type { RequestSummary } from '../lib/derive'

/**
 * Whether reuse is getting more expensive.
 *
 * The occupancy panel answers what the cache is holding; this one answers whether holding it is
 * still paying. Both readings are interval counts, so they are drawn as bands spanning the window
 * each sample covers, exactly like throughput.
 */
export function ChurnPanel({
  records,
  summary,
  replay,
}: {
  records: ThroughputRecord[]
  summary: RequestSummary
  replay: boolean
}) {
  const churn = summarizeChurn(records)
  const waste = summary.coverageWaste
  const restores = churn.restores.l1 + churn.restores.l2 + churn.restores.l3

  if (records.length === 0) {
    return (
      <Panel title="Cache churn" hint="churn" className="panel--wide">
        <Empty>{replay ? 'no throughput records in the loaded log' : 'no samples yet'}</Empty>
      </Panel>
    )
  }

  // A pinned domain keeps the two charts readable against each other: an eviction spike and the
  // restores it caused have to line up in time for either to mean anything.
  const first = records[0]!
  const last = records[records.length - 1]!
  const domain: [number, number] = [
    first.timestamp_unix_ms - first.interval_seconds * 1000,
    last.timestamp_unix_ms,
  ]

  return (
    <Panel
      title="Cache churn"
      hint="churn"
      className="panel--wide"
      note={`${count(restores)} restores · ${count(churn.evictions)} evictions`}
    >
      <div className="stat-row">
        <Stat
          value={restores === 0 ? '—' : percent(churn.importShare)}
          label="imported reuse"
          hint="importShare"
          tone={churn.importShare > 0.5 ? 'warning' : 'neutral'}
        />
        <Stat
          value={count(churn.lost)}
          label="lost sessions"
          hint="lostSessions"
          tone={churn.lost > 0 ? 'danger' : 'neutral'}
        />
        <Stat
          value={waste.requests === 0 ? '—' : percent(waste.prefillShare)}
          label="recomputed coverage"
          hint="coverageWaste"
          tone={waste.prefillShare > 0.1 ? 'danger' : 'neutral'}
        />
        <Stat
          value={count(churn.deferrals)}
          label="deferrals"
          hint="deferrals"
          tone={churn.deferrals > 0 ? 'warning' : 'neutral'}
        />
        <Stat
          value={count(churn.superseded)}
          label="superseded"
          hint="superseded"
          tone={churn.superseded > 0 ? 'warning' : 'neutral'}
        />
      </div>

      <BandChart
        label="Restores per interval, by the tier that served them"
        domain={domain}
        series={[
          {
            name: 'l1 resident',
            bands: bands(records, (r) => r.continuation_cache.tiers.delta_l1_restore_successes),
            color: CHART.accent,
          },
          {
            name: 'l2 host',
            bands: bands(records, (r) => r.continuation_cache.tiers.delta_l2_restore_successes),
            color: CHART.blue,
          },
          {
            name: 'l3 disk',
            bands: bands(records, (r) => r.continuation_cache.tiers.delta_l3_restore_successes),
            color: CHART.violet,
          },
        ]}
        unit="restores"
        stack
        integer
        legend={
          <Legend
            items={[
              { label: 'l1 resident', color: CHART.accent, hint: GLOSSARY.tierL1.body },
              { label: 'l2 host', color: CHART.blue, hint: GLOSSARY.tierL2.body },
              { label: 'l3 disk', color: CHART.violet, hint: GLOSSARY.tierL3.body },
            ]}
          />
        }
        caption={
          <>
            drift down the tiers is the churn signal <Info k="importShare" />
          </>
        }
      />

      <BandChart
        label="Retained lanes losing residency per interval"
        domain={domain}
        series={[
          {
            name: 'demoted',
            bands: bands(records, (r) => r.continuation_cache.occupancy.delta_l1_demotions),
            color: CHART.blue,
          },
          {
            name: 'lost',
            bands: bands(
              records,
              (r) =>
                r.continuation_cache.occupancy.delta_l1_evictions -
                r.continuation_cache.occupancy.delta_l1_demotions,
            ),
            color: CHART.danger,
          },
        ]}
        unit="lanes"
        stack
        integer
        legend={
          <Legend
            items={[
              { label: 'demoted', color: CHART.blue, hint: GLOSSARY.evictions.body },
              { label: 'lost', color: CHART.danger, hint: GLOSSARY.lostSessions.body },
            ]}
          />
        }
        caption={
          <>
            interval counts, not snapshots <Info k="intervalBands" />
          </>
        }
      />

      <p className="panel__footnote">
        {restores === 0 ? (
          'No restores in this window, so there is no tier mix to read yet.'
        ) : (
          <>
            <Term k="importShare">{percent(churn.importShare)} of reuse</Term> was imported rather
            than found resident ({count(churn.restores.l1)} L1 · {count(churn.restores.l2)} L2 ·{' '}
            {count(churn.restores.l3)} L3).{' '}
            {churn.lost === 0
              ? 'Every evicted session was handed to a lower tier, so none had to be recomputed.'
              : `${count(churn.lost)} of ${count(churn.evictions)} evicted sessions ${churn.lost === 1 ? 'was' : 'were'} dropped with no handoff.`}
          </>
        )}{' '}
        {churn.l2Evictions + churn.l3Evictions > 0 ? (
          <>
            <Term k="tierEvictions">Capacity evictions</Term>: {count(churn.l2Evictions)} from L2 (
            {bytes(churn.l2EvictedBytes)}) · {count(churn.l3Evictions)} from L3 (
            {bytes(churn.l3EvictedBytes)}).
          </>
        ) : null}{' '}
        {waste.requests > 0 ? (
          <>
            <Term k="coverageWaste">{count(waste.tokens)} prompt tokens</Term> across{' '}
            {count(waste.requests)} request
            {waste.requests === 1 ? '' : 's'} were prefilled despite preflight agreeing the cache
            held them — {percent(waste.prefillShare)} of all computed prefill.
          </>
        ) : null}
      </p>
    </Panel>
  )
}
