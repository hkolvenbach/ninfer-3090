// Derived request analytics.
//
// These aggregations are a port of the maintainer script `cache_health.py`, which is the checked
// reference for their definitions: same percentile rule, same denominators, same partitioning of
// misses and restore failures. `derive.test.ts` pins the agreement, so the dashboard and the
// script cannot drift into reporting different numbers for the same log.

import { CHART } from './palette'
import type { ContinuationSource, RequestDoneRecord } from './records'

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
