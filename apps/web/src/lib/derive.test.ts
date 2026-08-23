import { expect, test } from 'bun:test'

import { percentile, summarizeByAdapter, summarizeRequests } from './derive'
import { parseRecordLine, type RequestDoneRecord } from './records'

// Fixture shaped like real schema-15 records: a mix of tier sources, one cold miss, one restore
// failure. Expected values below were produced by the reference implementation
// (`cache_health.py`) over this same input, so a change here that drifts from the script fails.
function done(
  id: number,
  values: {
    prompt: number
    computed: number
    generated: number
    source: 'none' | 'l1' | 'l2' | 'l3'
    missReason?: string
    restoreFailure?: string
    reusePath?: string
    queue: number
    restore: number
    prefill: number
    decode: number
    ttft: number
    restoredTokens?: number
    restoredBytes?: number
    restoreMicroseconds?: number
    preflightMicroseconds?: number
    drafted?: number
    accepted?: number
    perPosition?: number[]
    adapter?: string
  },
): RequestDoneRecord {
  return {
    event: 'request_done',
    schema_version: 15,
    server_instance_id: 'serve-test-1',
    timestamp_unix_ms: 1_700_000_000_000 + id * 1000,
    artifact_type: 'ninfer_serve_request_log',
    request: {
      request_id: id,
      x_request_id: `req_${id}`,
      protocol: 'openai_responses',
      model: values.adapter ? `qwen3.8-27b-${values.adapter}` : 'qwen3.8-27b',
      adapter: values.adapter ?? '',
      stream: true,
      message_count: 4,
      tool_count: 0,
      has_tool_history: false,
      enable_thinking: true,
      requested_output_tokens: 4096,
      prompt_cache_key_digest: '',
    },
    result: {
      prompt_tokens: values.prompt,
      completion_tokens: values.generated,
      computed_prefill_tokens: values.computed,
      prefix_cache_hit_tokens: values.prompt - values.computed,
      prefix_reuse_path: values.reusePath ?? 'restore_turn_checkpoint',
      finish_reason: 'stop_token',
      tool_call_count: 0,
    },
    timings_seconds: {
      prepare: 0,
      queue: values.queue,
      restore: values.restore,
      prefill: values.prefill,
      decode: values.decode,
      publish: 0,
      vision: 0,
      ttft: values.ttft,
      total: values.ttft + values.decode,
    },
    continuation_cache: {
      source: values.source,
      alias_kind: 'routed_session',
      final_miss_reason: values.missReason ?? 'none',
      restore_failure: values.restoreFailure ?? 'none',
      lookup_microseconds: 0,
      preflight_microseconds: values.preflightMicroseconds ?? 0,
      restore_microseconds: values.restoreMicroseconds ?? 0,
      restored_tokens: values.restoredTokens ?? 0,
      restored_bytes: values.restoredBytes ?? 0,
      destructive_rollback: false,
      completion_publication_queued: true,
    },
    speculative:
      values.drafted === undefined
        ? undefined
        : {
            backend: 'mtp',
            draft_window: 3,
            rounds: 100,
            drafted_tokens: values.drafted,
            accepted_tokens: values.accepted ?? 0,
            fallback_steps: 0,
            accepted_per_position: values.perPosition ?? [],
          },
  }
}

const FIXTURE: RequestDoneRecord[] = [
  done(1, {
    prompt: 1000,
    computed: 100,
    generated: 200,
    source: 'l2',
    queue: 1,
    restore: 0.5,
    prefill: 0.25,
    decode: 2,
    ttft: 1.75,
    restoredTokens: 900,
    restoredBytes: 9_000_000,
    restoreMicroseconds: 500_000,
    preflightMicroseconds: 100_000,
    drafted: 300,
    accepted: 200,
    perPosition: [120, 50, 30],
  }),
  done(2, {
    prompt: 2000,
    computed: 2000,
    generated: 400,
    source: 'none',
    missReason: 'no_lane',
    reusePath: 'full_reset',
    queue: 4,
    restore: 0,
    prefill: 1,
    decode: 4,
    ttft: 5,
    drafted: 600,
    accepted: 300,
    perPosition: [180, 80, 40],
  }),
  done(3, {
    adapter: 'caveman',
    prompt: 4000,
    computed: 400,
    generated: 5000,
    source: 'l3',
    queue: 2,
    restore: 1.5,
    prefill: 0.5,
    decode: 50,
    ttft: 4,
    restoredTokens: 3600,
    restoredBytes: 36_000_000,
    restoreMicroseconds: 1_500_000,
    preflightMicroseconds: 300_000,
    drafted: 900,
    accepted: 600,
    perPosition: [400, 150, 50],
  }),
  done(4, {
    prompt: 500,
    computed: 0,
    generated: 50,
    source: 'l1',
    queue: 0.5,
    restore: 0,
    prefill: 0,
    decode: 0.5,
    ttft: 0.5,
    restoredTokens: 500,
    restoredBytes: 5_000_000,
    restoreMicroseconds: 10_000,
    preflightMicroseconds: 5_000,
  }),
  done(5, {
    adapter: 'caveman',
    prompt: 3000,
    computed: 3000,
    generated: 100,
    source: 'none',
    missReason: 'not_attempted',
    restoreFailure: 'kv_reservation',
    reusePath: 'full_reset',
    queue: 3,
    restore: 0,
    prefill: 1.5,
    decode: 1,
    ttft: 4.5,
  }),
]

test('percentile matches the reference truncating nearest-rank rule', () => {
  const values = [1, 2, 3, 4, 5]
  // int(q * len) indexes into the sorted array: 0.5 * 5 -> 2, 0.9 * 5 -> 4.
  expect(percentile(values, 0.5)).toBe(3)
  expect(percentile(values, 0.9)).toBe(5)
  expect(percentile(values, 0)).toBe(1)
  expect(percentile(values, 1)).toBe(5)
  expect(percentile([], 0.5)).toBe(0)
})

test('request summary reproduces the reference aggregations', () => {
  const summary = summarizeRequests(FIXTURE)

  expect(summary.count).toBe(5)

  // Sorted prompts [500, 1000, 2000, 3000, 4000]: p50 -> index 2, p90 -> index 4.
  expect(summary.promptTokens.p50).toBe(2000)
  expect(summary.promptTokens.p90).toBe(4000)
  expect(summary.promptTokens.max).toBe(4000)

  expect(summary.totalPromptTokens).toBe(10500)
  expect(summary.totalComputedPrefill).toBe(5500)
  expect(summary.prefillAvoided).toBeCloseTo(1 - 5500 / 10500, 12)

  expect(summary.generated.max).toBe(5000)
  expect(summary.overLongGenerations).toBe(1)

  expect(summary.bySource).toEqual({ l2: 1, none: 2, l3: 1, l1: 1 })
  expect(summary.byMissReason).toEqual({ no_lane: 1, not_attempted: 1 })
  expect(summary.byRestoreFailure).toEqual({ kv_reservation: 1 })
  expect(summary.byReusePath).toEqual({ restore_turn_checkpoint: 3, full_reset: 2 })
  expect(summary.byFinishReason).toEqual({ stop_token: 5 })

  // Sorted TTFT [0.5, 1.75, 4, 4.5, 5].
  expect(summary.ttft.p50).toBe(4)
  expect(summary.ttft.p90).toBe(5)

  // Shares are over summed TTFT (15.75), not per-request means.
  expect(summary.ttftShare.queue).toBeCloseTo(10.5 / 15.75, 12)
  expect(summary.ttftShare.restore).toBeCloseTo(2 / 15.75, 12)
  expect(summary.ttftShare.prefill).toBeCloseTo(3.25 / 15.75, 12)

  expect(summary.restores.count).toBe(3)
  expect(summary.restores.totalTokens).toBe(5000)
  expect(summary.restores.totalBytes).toBe(50_000_000)
  expect(summary.restores.meanSeconds).toBeCloseTo(0.67, 10)
  expect(summary.restores.meanPreflightSeconds).toBeCloseTo(0.135, 10)

  expect(summary.speculative.drafted).toBe(1800)
  expect(summary.speculative.accepted).toBe(1100)
  expect(summary.speculative.acceptRate).toBeCloseTo(1100 / 1800, 12)
  expect(summary.speculative.perPosition).toEqual([700, 280, 120])
})

test('an empty window summarizes without dividing by zero', () => {
  const summary = summarizeRequests([])
  expect(summary.count).toBe(0)
  expect(summary.prefillAvoided).toBe(0)
  expect(summary.ttftShare).toEqual({ queue: 0, restore: 0, prefill: 0 })
  expect(summary.speculative.acceptRate).toBe(0)
})

test('a torn trailing line is skipped rather than throwing', () => {
  // A reader tailing a file the server is still appending to sees a partial final record.
  expect(parseRecordLine('{"event":"throughput","schema_ver')).toBeNull()
  expect(parseRecordLine('')).toBeNull()
  expect(parseRecordLine('   ')).toBeNull()
  expect(parseRecordLine('{"not":"a record"}')).toBeNull()
  expect(parseRecordLine('{"event":"request_start","request":{}}')).not.toBeNull()
})

test('adapter grouping reproduces the reference per-adapter aggregations', () => {
  const usage = summarizeByAdapter(FIXTURE)

  // Base first, then adapters by descending request count.
  expect(usage.map((entry) => entry.name)).toEqual(['', 'caveman'])

  const base = usage[0]!
  expect(base.summary.count).toBe(3)
  expect(base.generatedTokens).toBe(650)
  expect(base.summary.ttft.p50).toBe(1.75)
  expect(base.summary.decodeTokensPerSecond.p50).toBeCloseTo(100, 12)
  expect(base.summary.prefillAvoided).toBeCloseTo(0.4, 12)
  expect(base.summary.speculative.acceptRate).toBeCloseTo(500 / 900, 12)

  const caveman = usage[1]!
  expect(caveman.summary.count).toBe(2)
  expect(caveman.generatedTokens).toBe(5100)
  expect(caveman.summary.ttft.p50).toBe(4.5)
  expect(caveman.summary.decodeTokensPerSecond.p50).toBeCloseTo(100, 12)
  expect(caveman.summary.prefillAvoided).toBeCloseTo(1 - 3400 / 7000, 12)
  expect(caveman.summary.speculative.acceptRate).toBeCloseTo(600 / 900, 12)

  // Groups partition the input: no request is counted twice or dropped.
  expect(usage.reduce((total, entry) => total + entry.summary.count, 0)).toBe(FIXTURE.length)
})

test('adapter grouping keys on the resolved adapter, not the requested model', () => {
  // The Anthropic route passes the client's own model string through and silently falls back to
  // the base weights when it does not resolve, so `model` cannot identify what actually ran.
  const record = structuredClone(FIXTURE[0]!)
  record.request.model = 'claude-sonnet-4'
  record.request.adapter = 'caveman'
  const usage = summarizeByAdapter([record])
  expect(usage.map((entry) => entry.name)).toEqual(['caveman'])
})
