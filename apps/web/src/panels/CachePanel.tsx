import { Legend, StackedBar } from '../components/charts'
import { CHART } from '../components/echart'
import { Term, Tooltip } from '../components/tooltip'
import { Empty, Meter, Panel, Pill, Stat } from '../components/ui'
import { SOURCE_COLOR, SOURCE_ORDER, type RequestSummary } from '../lib/derive'
import { bytes, count, percent } from '../lib/format'
import { GLOSSARY, type GlossaryKey } from '../lib/glossary'
import type { CacheTelemetry } from '../lib/telemetry'

const TIER_COLOR = { l1: CHART.accent, l2: CHART.blue, l3: CHART.violet } as const

// Explanations for each miss reason the engine can report, so the table is readable without
// cross-referencing the continuation-cache document.
const MISS_REASON: Record<string, string> = {
  not_attempted: 'No candidate was evaluated at all, usually a cold alias on a first-turn prompt.',
  no_lane: 'A candidate existed but no lane was free to import it into.',
  restore_failed: 'An import was attempted and failed; the prompt fell back to prefill.',
  not_deeper: 'The candidate did not extend the lane past what it already held, so it was skipped.',
  preflight_rejected: 'The candidate failed compatibility preflight against the running engine.',
  disabled: 'Continuation reuse is switched off for this request or this server.',
  no_alias: 'The request carried no session or stable-prefix alias to look up.',
  entry_unavailable_or_corrupt: 'The catalog named an entry that could not be read back.',
  rollback_conflict: 'A concurrent publication invalidated the candidate mid-admission.',
}

function Tier({
  name,
  hint,
  entries,
  used,
  capacity,
  color,
}: {
  name: string
  hint: GlossaryKey
  entries: number
  used: number
  capacity: number
  color: string
}) {
  const fraction = capacity === 0 ? 0 : used / capacity
  const tone = fraction > 0.95 ? 'danger' : fraction > 0.85 ? 'warning' : 'neutral'
  return (
    <div className="cache__tier">
      <div className="cache__tier-head">
        <span className="cache__tier-name">
          <Term k={hint}>{name}</Term>
        </span>
        <span className="cache__tier-fill">
          {bytes(used)} / {bytes(capacity)}
          <em>{percent(fraction)}</em>
        </span>
      </div>
      <Meter
        fraction={fraction}
        color={tone === 'danger' ? 'var(--danger)' : tone === 'warning' ? 'var(--warning)' : color}
      />
      <div className="cache__tier-foot">{count(entries)} entries</div>
    </div>
  )
}

export function CachePanel({
  cache,
  summary,
  replay,
}: {
  cache: CacheTelemetry | undefined
  summary: RequestSummary
  replay: boolean
}) {
  if (!cache) {
    return (
      <Panel title="Continuation cache" hint="promptSource">
        <Empty>{replay ? 'no throughput record in the loaded log' : 'engine not attached'}</Empty>
      </Panel>
    )
  }

  const saturated = (['l1', 'l2', 'l3'] as const).filter(
    (tier) =>
      cache[tier].capacity_bytes > 0 && cache[tier].bytes / cache[tier].capacity_bytes > 0.95,
  )
  const curtailed = cache.kv_growth.curtailed
  const lookups = cache.lookup_hits + cache.lookup_misses

  const sourceSegments = SOURCE_ORDER.filter((source) => (summary.bySource[source] ?? 0) > 0).map(
    (source) => ({
      label: source,
      value: summary.bySource[source] ?? 0,
      color: SOURCE_COLOR[source]!,
      hint:
        source === 'none'
          ? 'Prompt was prefilled from zero: no tier could serve it.'
          : GLOSSARY[source === 'l1' ? 'tierL1' : source === 'l2' ? 'tierL2' : 'tierL3'].body,
      display: `${summary.bySource[source] ?? 0} of ${summary.count} requests`,
    }),
  )

  return (
    <Panel
      title="Continuation cache"
      hint="promptSource"
      note={
        saturated.length > 0 ? (
          <Pill tone="warning">{saturated.join(' · ')} at capacity</Pill>
        ) : undefined
      }
    >
      <div className="stat-row">
        <Stat
          value={percent(summary.prefillAvoided)}
          label="prefill avoided"
          hint="prefillAvoided"
          tone={summary.prefillAvoided > 0.5 ? 'accent' : 'neutral'}
        />
        <Stat value={count(cache.restore_successes)} label="restores" hint="restores" />
        <Stat
          value={count(cache.restore_failures)}
          label="restore fails"
          hint="restoreFails"
          tone={cache.restore_failures > 0 ? 'danger' : 'neutral'}
        />
        {/* Catalog lookups are the L2/L3 path only: a resident L1 lane is matched directly and
            never consults the alias catalog, so this reads 0 on an all-L1 workload. */}
        <Stat
          value={lookups === 0 ? '—' : percent(cache.lookup_hits / lookups)}
          label="catalog hit"
          hint="catalogHit"
        />
        <Stat
          value={count(curtailed)}
          label="kv curtailed"
          hint="kvCurtailed"
          tone={curtailed > 0 ? 'danger' : 'neutral'}
        />
      </div>

      <div className="cache__tiers">
        <Tier
          name="L1 vram"
          hint="tierL1"
          entries={cache.l1.entries}
          used={cache.l1.bytes}
          capacity={cache.l1.capacity_bytes}
          color={TIER_COLOR.l1}
        />
        <Tier
          name="L2 host"
          hint="tierL2"
          entries={cache.l2.entries}
          used={cache.l2.bytes}
          capacity={cache.l2.capacity_bytes}
          color={TIER_COLOR.l2}
        />
        <Tier
          name="L3 disk"
          hint="tierL3"
          entries={cache.l3.entries}
          used={cache.l3.bytes}
          capacity={cache.l3.capacity_bytes}
          color={TIER_COLOR.l3}
        />
      </div>

      {summary.count === 0 ? (
        <Empty>no completed requests in the retained window</Empty>
      ) : (
        <div className="cache__sources">
          <div className="eyebrow">
            <Term k="promptSource">where prompts came from</Term> ({summary.count} requests)
          </div>
          <StackedBar segments={sourceSegments} />
          <Legend
            items={sourceSegments.map((s) => ({ label: s.label, color: s.color, hint: s.hint }))}
          />
          {Object.keys(summary.byMissReason).length > 0 ? (
            <table className="table">
              <thead>
                <tr>
                  <th>
                    <Term k="missReason">miss reason</Term>
                  </th>
                  <th className="numeric">n</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.byMissReason)
                  .sort((a, b) => b[1] - a[1])
                  .map(([reason, n]) => (
                    <tr key={reason}>
                      <td>
                        {MISS_REASON[reason] ? (
                          <Tooltip title={reason} body={MISS_REASON[reason]} className="tip--term">
                            {reason}
                          </Tooltip>
                        ) : (
                          reason
                        )}
                      </td>
                      <td className="numeric emphasis">{n}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : null}
        </div>
      )}

      <p className="panel__footnote">
        <Term k="evictions">L1 evictions</Term> {count(cache.l1.evictions)} · demotions{' '}
        {count(cache.l1.demotions)} · <Term k="kvGrowth">kv growth</Term> attempts{' '}
        {count(cache.kv_growth.attempts)} with {count(cache.kv_growth.forced_spills)} forced spills.
        Curtailed requests ended early at <code>length</code> because no pages were found. Catalog
        hit counts the L2/L3 alias lookup only — a prompt matched against a resident L1 lane never
        reaches it.
      </p>
    </Panel>
  )
}
