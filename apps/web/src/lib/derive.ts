// Derived request analytics.
//
// These aggregations are a port of the maintainer script `cache_health.py`, which is the checked
// reference for their definitions: same percentile rule, same denominators, same partitioning of
// misses and restore failures. `derive.test.ts` pins the agreement, so the dashboard and the
// script cannot drift into reporting different numbers for the same log.

import { CHART } from './palette'
import type { ContinuationSource, RequestDoneRecord, ThroughputRecord } from './records'

/**
 * Nearest-rank percentile with truncating index selection. Deliberately identical to the
 * reference script's `s[min(len(s) - 1, int(q * len(s)))]` rather than an interpolating
 * definition, because the two disagree on small samples and the script is the oracle.
 */
export function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.trunc(quantile * sorted.length))
  return sorted[index]!
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) return 0
  return values.reduce((total, value) => total + value, 0) / values.length
}

export function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0)
}

/** A counter a replayed log predates reads as zero, never as `NaN` in someone's total. */
function counter(value: number | undefined): number {
  return value ?? 0
}

function tally<T extends string>(values: readonly T[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1
  return counts
}

export interface Spread {
  p50: number
  p90: number
  p99: number
  max: number
  mean: number
}

function spread(values: readonly number[]): Spread {
  return {
    p50: percentile(values, 0.5),
    p90: percentile(values, 0.9),
    p99: percentile(values, 0.99),
    max: values.length === 0 ? 0 : Math.max(...values),
    mean: mean(values),
  }
}

/**
 * Prefill spent on state the cache demonstrably held.
 *
 * `deepest_candidate_agreement` is the deepest prefix any preflighted candidate agreed with, so a
 * request that still recomputed from zero proves the engine had that much of the prompt in hand
 * and prefilled anyway. Waste is measured beyond what the lane already reused
 * (`prefix_cache_hit_tokens`), which is what makes `not_deeper` - where the resident frontier
 * already covered the agreement - correctly score as no waste at all.
 *
 * `null` agreement means nothing was preflighted, so there is no evidence either way and the
 * request is excluded rather than counted as clean.
 */
export interface CoverageWaste {
  /** Requests that recomputed a prefix the cache had agreed with. */
  requests: number
  /** Tokens recomputed despite that agreement. */
  tokens: number
  /** Share of completed requests affected. */
  requestShare: number
  /** Share of all computed prefill that this waste accounts for. */
  prefillShare: number
}

export interface RestoreSummary {
  count: number
  meanTokens: number
  totalTokens: number
  meanBytes: number
  totalBytes: number
  meanSeconds: number
  p90Seconds: number
  meanPreflightSeconds: number
}

export interface SpeculativeSummary {
  drafted: number
  accepted: number
  /** Accepted fraction of drafted tokens, the headline MTP figure. */
  acceptRate: number
  /** Acceptance count per draft position, summed across requests. */
  perPosition: number[]
  fallbackSteps: number
}

export interface RequestSummary {
  count: number
  promptTokens: Spread
  computedPrefill: Spread
  generated: Spread
  totalPromptTokens: number
  totalComputedPrefill: number
  /** Fraction of prompt tokens that prefix reuse kept out of prefill entirely. */
  prefillAvoided: number
  overLongGenerations: number
  bySource: Record<string, number>
  byMissReason: Record<string, number>
  byRestoreFailure: Record<string, number>
  byReusePath: Record<string, number>
  byFinishReason: Record<string, number>
  ttft: Spread
  /** Share of summed TTFT attributable to each phase. These are the actionable latency terms. */
  ttftShare: { queue: number; restore: number; prefill: number }
  coverageWaste: CoverageWaste
  restores: RestoreSummary
  speculative: SpeculativeSummary
  decodeTokensPerSecond: Spread
}

export function summarizeRequests(records: readonly RequestDoneRecord[]): RequestSummary {
  const prompt = records.map((record) => record.result.prompt_tokens)
  const computed = records.map((record) => record.result.computed_prefill_tokens)
  const generated = records.map((record) => record.result.completion_tokens)
  const ttft = records.map((record) => record.timings_seconds.ttft)

  const totalPrompt = sum(prompt)
  const totalComputed = sum(computed)
  const totalTtft = sum(ttft)

  const restored = records.filter((record) => record.continuation_cache.source !== 'none')
  const restoreSeconds = restored.map(
    (record) => record.continuation_cache.restore_microseconds / 1e6,
  )

  const perPosition: number[] = []
  let drafted = 0
  let accepted = 0
  let fallbackSteps = 0
  for (const record of records) {
    const speculative = record.speculative
    if (!speculative) continue
    drafted += speculative.drafted_tokens
    accepted += speculative.accepted_tokens
    fallbackSteps += speculative.fallback_steps
    speculative.accepted_per_position.forEach((value, index) => {
      perPosition[index] = (perPosition[index] ?? 0) + value
    })
  }

  // Tokens the cache agreed with and the engine recomputed anyway, counted only beyond what the
  // lane had already reused. A restore that happened is not waste, and an agreement the resident
  // frontier already covered is not waste either.
  let wasteTokens = 0
  let wasteRequests = 0
  for (const record of records) {
    if (record.continuation_cache.source !== 'none') continue
    const agreement = record.continuation_cache.deepest_candidate_agreement
    if (agreement === null || agreement === undefined) continue
    const recomputed = agreement - record.result.prefix_cache_hit_tokens
    if (recomputed <= 0) continue
    wasteTokens += recomputed
    wasteRequests += 1
  }

  // Only requests that actually decoded contribute a rate; a fully restored prompt that emitted
  // one token in near-zero time would otherwise dominate the spread with a meaningless value.
  const decodeRates = records
    .filter((record) => record.timings_seconds.decode > 0.01)
    .map((record) => record.result.completion_tokens / record.timings_seconds.decode)

  return {
    count: records.length,
    promptTokens: spread(prompt),
    computedPrefill: spread(computed),
    generated: spread(generated),
    totalPromptTokens: totalPrompt,
    totalComputedPrefill: totalComputed,
    prefillAvoided: totalPrompt === 0 ? 0 : 1 - totalComputed / totalPrompt,
    overLongGenerations: generated.filter((value) => value > 4096).length,
    bySource: tally(records.map((record) => record.continuation_cache.source)),
    byMissReason: tally(
      records
        .filter((record) => record.continuation_cache.source === 'none')
        .map((record) => record.continuation_cache.final_miss_reason),
    ),
    byRestoreFailure: tally(
      records
        .map((record) => record.continuation_cache.restore_failure)
        .filter((value) => value !== 'none'),
    ),
    byReusePath: tally(records.map((record) => record.result.prefix_reuse_path)),
    byFinishReason: tally(records.map((record) => record.result.finish_reason)),
    ttft: spread(ttft),
    ttftShare: {
      queue: totalTtft === 0 ? 0 : sum(records.map((r) => r.timings_seconds.queue)) / totalTtft,
      restore: totalTtft === 0 ? 0 : sum(records.map((r) => r.timings_seconds.restore)) / totalTtft,
      prefill: totalTtft === 0 ? 0 : sum(records.map((r) => r.timings_seconds.prefill)) / totalTtft,
    },
    coverageWaste: {
      requests: wasteRequests,
      tokens: wasteTokens,
      requestShare: records.length === 0 ? 0 : wasteRequests / records.length,
      prefillShare: totalComputed === 0 ? 0 : wasteTokens / totalComputed,
    },
    restores: {
      count: restored.length,
      meanTokens: mean(restored.map((r) => r.continuation_cache.restored_tokens)),
      totalTokens: sum(restored.map((r) => r.continuation_cache.restored_tokens)),
      meanBytes: mean(restored.map((r) => r.continuation_cache.restored_bytes)),
      totalBytes: sum(restored.map((r) => r.continuation_cache.restored_bytes)),
      meanSeconds: mean(restoreSeconds),
      p90Seconds: percentile(restoreSeconds, 0.9),
      meanPreflightSeconds: mean(
        restored.map((r) => r.continuation_cache.preflight_microseconds / 1e6),
      ),
    },
    speculative: {
      drafted,
      accepted,
      acceptRate: drafted === 0 ? 0 : accepted / drafted,
      perPosition,
      fallbackSteps,
    },
    decodeTokensPerSecond: spread(decodeRates),
  }
}

/** Per-adapter usage. `name` is `""` for the base model, which is not an adapter. */
export interface AdapterUsage {
  name: string
  summary: RequestSummary
  generatedTokens: number
}

/**
 * Groups completed requests by the adapter that served them.
 *
 * `request.adapter` is the authoritative field, not `request.model`: on the Anthropic route the
 * model string is whatever the client sent and an unknown value silently falls back to the base
 * model rather than 404ing, so only the resolved adapter name identifies what actually ran.
 *
 * Each group is summarized with the same `summarizeRequests` used for the global figures, so a
 * per-adapter number and its global counterpart are computed identically.
 */
export function summarizeByAdapter(records: readonly RequestDoneRecord[]): AdapterUsage[] {
  const groups = new Map<string, RequestDoneRecord[]>()
  for (const record of records) {
    const name = record.request.adapter ?? ''
    const bucket = groups.get(name)
    if (bucket === undefined) groups.set(name, [record])
    else bucket.push(record)
  }
  return (
    [...groups.entries()]
      .map(([name, group]) => ({
        name,
        summary: summarizeRequests(group),
        generatedTokens: sum(group.map((record) => record.result.completion_tokens)),
      }))
      // Base first, then adapters by traffic. The base row is the comparison every other row is
      // read against, so it does not compete for position on request count.
      .sort((a, b) => (a.name === '' ? -1 : b.name === '' ? 1 : b.summary.count - a.summary.count))
  )
}

/**
 * Cache churn over the retained throughput window.
 *
 * Eviction on its own is a cache doing its job, so none of these are pathological in isolation.
 * The two readings that matter are `lost`, which is state discarded with no handoff and therefore
 * guaranteed to be recomputed if it is wanted again, and `importShare`, which is the fraction of
 * reuse that had to come from host or disk instead of resident VRAM. A working set that outgrows
 * L1 keeps its hit rate and quietly starts paying an import on every turn; that shift shows up
 * here and nowhere else.
 *
 * Summed from interval deltas rather than differenced endpoints, so a server restart inside the
 * window cannot turn a counter reset into a negative or absurd reading. A counter a replayed log
 * predates reads as zero rather than poisoning the whole summary with `NaN`.
 */
export interface ChurnSummary {
  evictions: number
  demotions: number
  /** Evicted with no publication ticket, so the session did not survive anywhere. */
  lost: number
  l2Evictions: number
  l3Evictions: number
  l2EvictedBytes: number
  l3EvictedBytes: number
  restores: { l1: number; l2: number; l3: number }
  /** Share of restores served from L2 or L3 rather than resident L1. */
  importShare: number
  /** Restores refused for shared-KV capacity with the candidate left live for a retry. */
  deferrals: number
  /** Publications that completed and were discarded because the alias had already moved on. */
  superseded: number
}

export function summarizeChurn(records: readonly ThroughputRecord[]): ChurnSummary {
  let evictions = 0
  let demotions = 0
  let l2Evictions = 0
  let l3Evictions = 0
  let l2EvictedBytes = 0
  let l3EvictedBytes = 0
  let l1 = 0
  let l2 = 0
  let l3 = 0
  let deferrals = 0
  let superseded = 0
  for (const record of records) {
    const cache = record.continuation_cache
    const occupancy = cache.occupancy
    evictions += occupancy.delta_l1_evictions
    demotions += occupancy.delta_l1_demotions
    l2Evictions += counter(occupancy.delta_l2_evictions)
    l3Evictions += counter(occupancy.delta_l3_evictions)
    l2EvictedBytes += counter(occupancy.delta_l2_evicted_bytes)
    l3EvictedBytes += counter(occupancy.delta_l3_evicted_bytes)
    l1 += cache.tiers.delta_l1_restore_successes
    l2 += cache.tiers.delta_l2_restore_successes
    l3 += cache.tiers.delta_l3_restore_successes
    deferrals += counter(cache.delta_restore_deferrals)
    superseded += counter(cache.delta_publication_superseded)
  }
  const restores = l1 + l2 + l3
  return {
    evictions,
    demotions,
    // A demotion is one kind of eviction, never an extra one, so this cannot go negative.
    lost: evictions - demotions,
    l2Evictions,
    l3Evictions,
    l2EvictedBytes,
    l3EvictedBytes,
    restores: { l1, l2, l3 },
    importShare: restores === 0 ? 0 : (l2 + l3) / restores,
    deferrals,
    superseded,
  }
}

export const SOURCE_ORDER: readonly ContinuationSource[] = ['l1', 'l2', 'l3', 'none']

/**
 * Tier colors, shared by the source breakdown and the cache occupancy panel.
 *
 * Literal values rather than `var()` because the same palette feeds ECharts, which renders to
 * canvas and cannot resolve custom properties.
 */
export const SOURCE_COLOR: Record<string, string> = {
  l1: CHART.accent,
  l2: CHART.blue,
  l3: CHART.violet,
  none: CHART.dim,
}
