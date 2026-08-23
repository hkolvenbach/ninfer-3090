// Record shapes, as emitted by src/serve/request_log.cpp (schema 15).
//
// GET /events streams these live and `--request-log-jsonl` appends the identical lines, so one
// set of types serves both the live dashboard and file replay. Only the fields the dashboard
// reads are declared; a record carries more, and unread fields are deliberately not mirrored
// here because every declaration is a second place to keep in step with the C++ formatter.
//
// Records are discriminated on `event`, never on `schema_version`, so an older log still
// replays: fields added by a later schema are declared optional and read as absent.

export interface RequestContext {
  request_id: number
  x_request_id: string
  protocol: string
  model: string
  adapter: string
  stream: boolean
  message_count: number
  tool_count: number
  has_tool_history: boolean
  enable_thinking: boolean
  requested_output_tokens: number
  prompt_cache_key_digest: string
}

/** Where a prompt's resident state came from. `none` means it was prefilled from zero. */
export type ContinuationSource = 'none' | 'l1' | 'l2' | 'l3'

export interface ContinuationDiagnostics {
  source: ContinuationSource
  alias_kind: string
  final_miss_reason: string
  restore_failure: string
  lookup_microseconds: number
  preflight_microseconds: number
  restore_microseconds: number
  restored_tokens: number
  restored_bytes: number
  destructive_rollback: boolean
  completion_publication_queued: boolean
}

/** Per-request phase durations. `ttft` is the quantity the queue/restore/prefill split explains. */
export interface RequestTimings {
  prepare: number
  queue: number
  restore: number
  prefill: number
  decode: number
  publish: number
  vision: number
  ttft: number
  total: number
}

export interface SpeculativeStats {
  backend: string
  draft_window: number
  rounds: number
  drafted_tokens: number
  accepted_tokens: number
  fallback_steps: number
  /** Acceptance count per draft position; its decay is the shape of the MTP accept curve. */
  accepted_per_position: number[]
}

/**
 * The resident adapter bank, as reported by `server_start` and `/telemetry`.
 *
 * Every registered adapter shares one rank by construction, so a single rank and one bank byte
 * count describe the whole set. `device_bytes` is the bank's own arena, which sits outside the
 * weights arena and is otherwise visible only as a reduction in free memory.
 */
export interface AdapterInventory {
  count: number
  names: string[]
  rank: number
  device_bytes: number
  file_bytes: number
  /** Served model ids, `/telemetry` only; `server_start` reports names alone. */
  model_ids?: string[]
}

interface RecordEnvelope {
  schema_version: number
  server_instance_id: string
  timestamp_unix_ms: number
  artifact_type: string
}

export interface ServerStartRecord extends RecordEnvelope {
  event: 'server_start'
  argv: string[]
  artifact: {
    path: string
    target: string
    weights_id: string
    size_bytes: number
    tensor_count: number
    load_seconds: number
    upload_seconds: number
  }
  engine: {
    device: number
    kv_cache: string
    kv_capacity: number
    kv_capacity_mode: string
    max_concurrency: number
    max_context: number
    max_pending_requests: number
    pending_timeout_ms: number
    prefill_chunk: number
    prefix_reuse: boolean
    prefix_checkpoint_policy: string
    speculative_backend: string
    speculative_draft_window: number
    cuda_graph: boolean
    vision: boolean
    log_stats_interval_ms: number
    continuation_cache: {
      tiers: string
      policy: string
      directory: string
      namespace: string
      l1_capacity_mib: number
      l2_capacity_mib: number
      l3_capacity_mib: number
      persist_min_tokens: number
    }
  }
  /** Resident LoRA bank, present from schema 15. Names appear even with no adapter traffic. */
  adapters?: AdapterInventory
  environment: {
    device: number
    gpu_name: string
    gpu_uuid: string
    total_device_memory_bytes: number
    compute_capability_major: number
    compute_capability_minor: number
    cuda_runtime_version: string
    cuda_driver_version: string
  }
  memory: Record<string, unknown> & { lora_bank_bytes?: number }
  server: {
    host: string
    port: number
    public_model_id: string
    cors_enabled: boolean
    api_key_configured: boolean
  }
}

export interface RequestStartRecord extends RecordEnvelope {
  event: 'request_start'
  request: RequestContext
}

export interface RequestDoneRecord extends RecordEnvelope {
  event: 'request_done'
  request: RequestContext
  result: {
    prompt_tokens: number
    completion_tokens: number
    computed_prefill_tokens: number
    prefix_cache_hit_tokens: number
    prefix_reuse_path: string
    finish_reason: string
    tool_call_count: number
  }
  timings_seconds: RequestTimings
  continuation_cache: ContinuationDiagnostics
  speculative?: SpeculativeStats
}

export interface RequestErrorRecord extends RecordEnvelope {
  event: 'request_error'
  request: RequestContext
  error: { message: string }
}

/**
 * One reporting interval. The reporter only emits when the interval had activity, so samples are
 * irregularly spaced and `interval_seconds` — not the gap between timestamps — is the authority
 * on the window a sample covers.
 */
export interface ThroughputRecord extends RecordEnvelope {
  event: 'throughput'
  interval_seconds: number
  throughput_tokens_per_second: { prefill: number; decode: number }
  tokens: { computed_prefill: number; committed_decode: number }
  decode_batch: { rounds: number; row_rounds: number; average_size: number | null }
  scheduler: {
    running: number
    prefilling: number
    decode_ready: number
    waiting: number
    admitted_requests: number
    delta_admitted_requests: number
    queue_seconds_total: number
    delta_queue_seconds: number
    rejected_overloaded: number
    rejected_queue_timeout: number
  }
  continuation_cache: {
    occupancy: {
      l1_entries: number
      l1_bytes: number
      l2_entries: number
      l2_bytes: number
      l3_entries: number
      l3_bytes: number
      l1_evictions: number
      l1_demotions: number
      kv_growth_attempts: number
      kv_growth_forced_spills: number
      kv_growth_curtailed: number
    }
    delta_lookup_hits: number
    delta_lookup_misses: number
    delta_restore_successes: number
    delta_restore_failures: number
    delta_restored_tokens: number
    delta_restored_bytes: number
    // Cumulative totals alongside the deltas. These are what let a replayed log reconstruct the
    // same cache view the live /telemetry snapshot reports.
    lookup_hits: number
    lookup_misses: number
    restore_successes: number
    restore_failures: number
    publication_successes: number
    publication_failures: number
    persistence_total: {
      queued: number
      coalesced: number
      successes: number
      failures: number
    }
    tiers: Record<string, number>
    miss_reasons: Record<string, number>
  }
}

export type EngineRecord =
  ServerStartRecord | RequestStartRecord | RequestDoneRecord | RequestErrorRecord | ThroughputRecord

export type RecordEvent = EngineRecord['event']

export function isEngineRecord(value: unknown): value is EngineRecord {
  if (typeof value !== 'object' || value === null) return false
  const event = (value as { event?: unknown }).event
  return (
    event === 'server_start' ||
    event === 'request_start' ||
    event === 'request_done' ||
    event === 'request_error' ||
    event === 'throughput'
  )
}

/**
 * Parses one JSONL line. Returns null for blank lines and for a torn trailing record, which a
 * reader tailing a file the server is still appending to will encounter.
 */
export function parseRecordLine(line: string): EngineRecord | null {
  const trimmed = line.trim()
  if (trimmed.length === 0) return null
  try {
    const parsed: unknown = JSON.parse(trimmed)
    return isEngineRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}
