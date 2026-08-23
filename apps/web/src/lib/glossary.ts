// Definitions for every term the dashboard shows.
//
// One place so a reading and its explanation cannot drift, and so the wording stays specific to
// this engine rather than generic inference vocabulary. Each entry says what the number measures,
// and where it is useful, what to conclude when it moves.

export interface Definition {
  title: string
  body: string
}

export const GLOSSARY = {
  // --- rates -------------------------------------------------------------------------------
  decodeRate: {
    title: 'Decode tokens/s (aggregate)',
    body: 'Tokens committed by decode rounds over the last interval, summed across every active lane — engine-wide, not per request. It rises with concurrency even when no individual request gets faster. The first token comes from prefill and is excluded; with MTP, accepted tokens count and rejected drafts do not.',
  },
  perSequenceRate: {
    title: 'Per-sequence decode rate',
    body: 'Aggregate decode divided by the mean decode batch: the rate one request experiences, because a sequence in a batch of b receives 1/b of what those rounds committed. This is the number to compare against a single-user benchmark; the aggregate is the number to compare against total server capacity.',
  },
  prefillRate: {
    title: 'Prefill tokens/s',
    body: 'Prompt tokens actually evaluated over the last interval. Tokens served from a reused prefix are excluded, so this measures real compute, not prompt size. Prefill and decode share one execution thread, so a high prefill rate usually coincides with a depressed decode rate.',
  },
  decodeBatch: {
    title: 'Average decode batch',
    body: 'Mean sequences advanced per decode round over the interval (round rows ÷ rounds). The engine forms one compact batch at every round boundary, so this rises toward the lane count when several requests decode concurrently. It is the multiplier between per-sequence and aggregate throughput: a value near 1 under load means requests are serialized behind prefill rather than batching.',
  },

  // --- scheduling --------------------------------------------------------------------------
  lanes: {
    title: 'Lanes',
    body: 'Running requests against --max-concurrency. A lane is a resident execution slot with its own KV pages. The count is fixed at startup between 1 and 8; there is no preemption, so an occupied lane is held until its request finishes.',
  },
  queued: {
    title: 'Queued requests',
    body: 'Requests admitted to the bounded FIFO but not yet given a lane. Ingress is bounded by --max-pending-requests; beyond that the server returns 429. A request that waits past --pending-timeout-ms is rejected with 503.',
  },
  meanQueue: {
    title: 'Mean queue delay',
    body: 'Cumulative time requests spent between submission and admission, divided by the number of admissions. This is the term that dominates time-to-first-token once the cache is working, and it responds to lane count and generation length, not to kernel speed.',
  },
  rejected: {
    title: 'Rejected requests',
    body: 'Requests refused before execution: the FIFO was full (429) or the admission deadline expired while waiting (503). A rejected request produces no request-log record, so this counter is the only place it appears.',
  },
  laneOccupancy: {
    title: 'Lane occupancy',
    body: 'Running requests as a fraction of the configured lane count. Sustained saturation with a non-empty queue means throughput is admission-bound, not compute-bound.',
  },
  ingressQueue: {
    title: 'Ingress queue',
    body: 'Waiting requests as a fraction of --max-pending-requests. Reaching the limit means new requests are being rejected with 429.',
  },

  // --- worker ------------------------------------------------------------------------------
  workerSplit: {
    title: 'Execution thread wall clock',
    body: 'How the single execution thread spent its time. Every unit runs behind one mutex, so a second spent in any of them is a second in which no other resident lane advances. Prefill share rising against decode is what starves a decoding request.',
  },
  workerDecode: {
    title: 'Decode',
    body: 'Time in decode rounds. Rounds are short, so a high share with a low decode rate means many small rounds rather than batched ones.',
  },
  workerPrefill: {
    title: 'Prefill',
    body: 'Time evaluating prompt chunks, sized by --prefill-chunk. Chunks are long relative to decode rounds, so prefill is the usual source of decode stalls.',
  },
  workerAdmission: {
    title: 'Admission',
    body: 'Time admitting requests to lanes, including continuation import. A large share here is cache restore work on the critical path, not scheduling overhead.',
  },
  workerPublish: {
    title: 'Publish',
    body: 'Time publishing continuation images so a later request can reuse the session. This is what populates L2 and queues L3 writes.',
  },
  workerUpkeep: {
    title: 'Upkeep',
    body: 'Periodic maintenance on the execution thread. Normally negligible.',
  },
  admissionSplit: {
    title: 'Admission decomposition',
    body: 'Admission time split into planning, continuation restore, and commit, over the scheduler iterations that entered the attempt. Most calls admit nothing, so the per-call figure is what determines how much of the execution thread admission consumes.',
  },

  // --- latency -----------------------------------------------------------------------------
  ttft: {
    title: 'Time to first token',
    body: 'Submission to first emitted token: queue wait, then continuation restore, then prefill. Reported from the engine, not from HTTP round-trip timing, so it excludes prompt preparation and vision encode.',
  },
  ttftSplit: {
    title: 'TTFT decomposition',
    body: 'Share of summed TTFT spent queueing for a lane, restoring cached state, and prefilling. Queue-dominated means requests are waiting, not computing — more lanes or shorter generations move it before any kernel work does. Prefill-dominated means prompts are genuinely being recomputed; check the prompt-source split.',
  },
  percentiles: {
    title: 'Percentiles',
    body: 'Nearest-rank over the retained request window, matching the maintainer cache_health.py rule. p50 is typical, p90 and p99 show the tail that a client actually notices.',
  },
  decodePerRequest: {
    title: 'Per-request decode rate',
    body: 'Generated tokens divided by that request’s own decode seconds. Not a partition of the aggregate: every lane in a round is charged the full round wall time, so these do not sum to the engine-wide rate. Read it as the speed that request experienced.',
  },

  // --- speculation -------------------------------------------------------------------------
  mtpAccept: {
    title: 'MTP acceptance',
    body: 'Fraction of speculatively drafted tokens the verifier accepted. Each accepted token is one the model did not have to decode serially, so acceptance translates almost directly into decode throughput. Rejected drafts cost the draft work but never change output.',
  },

  // --- cache -------------------------------------------------------------------------------
  prefillAvoided: {
    title: 'Prefill avoided',
    body: 'Share of prompt tokens across completed requests that never reached prefill because a resident or restored prefix already covered them. The single clearest measure of whether the continuation cache is doing its job.',
  },
  promptSource: {
    title: 'Prompt source',
    body: 'Which tier served each prompt. L1 is a session still resident in VRAM and costs nothing to reuse. L2 is a host-RAM image copied back to the device. L3 is read from disk. None means the prompt was prefilled from zero.',
  },
  tierL1: {
    title: 'L1 — resident VRAM',
    body: 'Sessions still held in a lane’s KV pages. Reuse requires no import at all, so an L1 hit is the cheapest possible path. Capacity is --continuation-cache-l1-mib; pressure shows up as evictions and demotions.',
  },
  tierL2: {
    title: 'L2 — host RAM',
    body: 'Continuation images in host memory. A hit costs a host-to-device transfer, which is fast but on the critical path of admission. Capacity is --continuation-cache-l2-mib.',
  },
  tierL3: {
    title: 'L3 — disk',
    body: 'Content-addressed images persisted under --continuation-cache-dir. A hit costs a disk read plus transfer, still far cheaper than re-prefilling a long prompt. Capacity is --continuation-cache-l3-mib; at capacity, older entries are evicted rather than new ones refused.',
  },
  catalogHit: {
    title: 'Catalog hit rate',
    body: 'Hit rate of the alias catalog lookup, which is the L2 and L3 path only. A prompt matched against a resident L1 lane never consults the catalog, so an all-L1 workload reads 0% here while still avoiding nearly all prefill.',
  },
  restores: {
    title: 'Restores',
    body: 'Imports that successfully brought cached state into a lane, across all tiers. Counted only when the restore was useful — deeper than the lane’s existing frontier.',
  },
  restoreFails: {
    title: 'Restore failures',
    body: 'Imports that were attempted and failed, attributed by cause: KV reservation exhausted, verification depth disagreement, segment inventory mismatch, metadata mismatch, decode or transfer error, or no lane able to accept the import. Failures fall back to prefill, so they cost latency but never correctness.',
  },
  missReason: {
    title: 'Miss reasons',
    body: 'Why a prompt ended up prefilling from zero. not_attempted means no candidate was evaluated (usually a cold alias); no_lane means nothing was free to import into; restore_failed means an import was tried and failed; not_deeper means the candidate added nothing over the lane’s current state.',
  },
  kvGrowth: {
    title: 'On-demand KV growth',
    body: 'A request reserves a bounded decode window at admission and acquires the rest as it generates. Attempts counts round boundaries that asked for more pages; forced spills counts retained sessions demoted to make room; curtailed counts requests that ended early at length because neither rung found pages.',
  },
  kvCurtailed: {
    title: 'Curtailed requests',
    body: 'Requests that stopped generating early because no KV pages could be found, reported to the client as finish reason length. Any non-zero value means output was truncated by capacity, not by the model.',
  },
  evictions: {
    title: 'L1 evictions and demotions',
    body: 'Retained lanes destroyed outright, versus demoted into L2 or L3 so their session survives. Demotions preserve reuse; evictions do not.',
  },
  reusePath: {
    title: 'Prefix reuse path',
    body: 'restore_turn_checkpoint means the prompt diverged mid-history and resumed from the nearest retained turn checkpoint. full_reset means the lane started from zero.',
  },
  sessionDigest: {
    title: 'Session digest',
    body: 'Stable identifier of a lane’s exact resident token ledger. Equal digests mean the identical session. Opaque to clients, but usable as an if_digest precondition on slot operations.',
  },
  checkpoints: {
    title: 'Turn checkpoints',
    body: 'Retained host snapshots a diverging prompt can rewind to, sized by --turn-checkpoints. Each holds a full state image, so they trade host memory for avoided re-prefill.',
  },
  slotReused: {
    title: 'Reused share',
    body: 'Portion of the lane’s prompt served from resident prefix rather than recomputed.',
  },
  retainedLane: {
    title: 'Retained lane',
    body: 'An idle lane still holding a resident session in VRAM. A matching prompt reuses it with no import at all; a non-matching one evicts or demotes it.',
  },

  // --- adapters ----------------------------------------------------------------------------
  adapterBank: {
    title: 'Adapter bank',
    body: 'All registered LoRA adapters, packed into one device arena at startup. Every adapter shares one rank, so the rank and per-site strides are kernel constants. The bank is resident for the process lifetime — there is no load, unload, or activation path, and selection is a per-row index at execution time.',
  },
  adapterVram: {
    title: 'Adapter VRAM',
    body: 'The bank lives outside the weights arena and is committed before KV capacity is resolved, so it silently reduces the KV budget. It is reported separately here because it belongs to neither the weights nor the KV figure.',
  },
  adapterUsage: {
    title: 'Per-adapter usage',
    body: 'Completed requests grouped by the adapter that actually served them. A registered adapter with no rows is resident and costing VRAM without carrying traffic.',
  },

  // --- GPU ---------------------------------------------------------------------------------
  gpuUtil: {
    title: 'SM utilization',
    body: 'Fraction of the sampling window in which at least one kernel was resident, from NVML. It says the GPU was busy, not that it was efficient — a memory-bound decode can report 100% while far from peak throughput.',
  },
  gpuMemBw: {
    title: 'Memory bandwidth utilization',
    body: 'Fraction of the window in which device memory was being read or written. Decode of a large model is memory-bound, so this sitting near 100% during decode is expected.',
  },
  gpuTemp: {
    title: 'Temperature',
    body: 'Core temperature. A 24 GB RTX 4090 under sustained prefill approaches its thermal limit and will reduce clocks; that is the documented way a prefill measurement gets misread as a kernel regression.',
  },
  gpuPower: {
    title: 'Power draw',
    body: 'Instantaneous draw against the enforced limit. Sitting at the limit means clocks are being capped by power, which appears as sw_power_cap in the throttle reasons.',
  },
  gpuClock: {
    title: 'SM clock',
    body: 'Current shader clock against the board maximum. A sustained gap with an active throttle reason explains a throughput drop that is not a code change.',
  },
  gpuThrottle: {
    title: 'Clock throttle reasons',
    body: 'Active NVML reasons the clocks are limited. gpu_idle and applications_clocks_setting are not faults and are not flagged. sw_power_cap, hw_slowdown, and the thermal reasons mean measured throughput is being limited by the board, not the implementation.',
  },
  boardMemory: {
    title: 'Board memory',
    body: 'Total VRAM in use on the device from NVML, including this process and anything else resident. Compare against the engine’s own arena budget to see whether another process is competing.',
  },

  // --- memory ------------------------------------------------------------------------------
  vramBudget: {
    title: 'VRAM budget',
    body: 'How the engine divided the board at startup. Arenas are reserved up front, so capacity — not current use — is what the device actually holds. Free is what remained after weights, KV, workspace, graph allowances, and any adapter bank. These are the engine’s own arenas; the board figure above also includes the CUDA context and any other process, so the two are not expected to match exactly.',
  },
  weightsArena: {
    title: 'Weights',
    body: 'Model weights uploaded once at load, in their registered storage format.',
  },
  kvArena: {
    title: 'KV arena',
    body: 'Reservation backing the paged KV cache and per-sequence state, sized from --kv-capacity and the selected --kv-dtype codec.',
  },
  kvPayload: {
    title: 'KV payload',
    body: 'Bytes of the KV arena currently holding real cache content, against the arena reserved for it.',
  },
  workspaceArena: {
    title: 'Workspace',
    body: 'Transient scratch for kernels. Peak use against capacity shows how much of the reservation is actually needed.',
  },
  cudaGraphs: {
    title: 'CUDA graphs',
    body: 'Memory captured graphs occupy against the allowance planned for them. Graphs require stable device addresses, which is why the allowance is reserved rather than allocated on demand.',
  },
  pageGroups: {
    title: 'KV page groups',
    body: 'Allocated page groups against the maximum the arena can address. This is the real capacity ceiling for concurrent context.',
  },
  textKv: {
    title: 'Text KV',
    body: 'KV cache for the main attention path.',
  },
  gdnState: {
    title: 'GDN state',
    body: 'Per-sequence recurrent state for the gated delta-net layers, held in FP32.',
  },
  mtpKv: {
    title: 'MTP KV',
    body: 'KV cache for the speculative draft head.',
  },

  // --- reading the charts ------------------------------------------------------------------
  intervalBands: {
    title: 'Why bars have different widths',
    body: 'Each bar covers its own reporting interval. The reporter skips intervals with no activity and folds the skipped time into the next sample, so a wide bar is idle time folded forward, not a long sustained rate.',
  },
  snapshotSeries: {
    title: 'Snapshots, not interval means',
    body: 'Lane and queue counts are read at the instant each report is emitted, unlike throughput which is averaged over the interval. They are drawn as points at their own timestamps so the chart does not claim an occupancy that was never measured.',
  },
  replayMode: {
    title: 'Replay',
    body: 'Showing a loaded request log instead of the live engine. Throughput, latency, per-request detail, and cache occupancy against configured capacity all come from the file. Board telemetry and live lane occupancy are sampled, never recorded, so those panels stay empty.',
  },
  eventStream: {
    title: 'Live data',
    body: 'Levels come from GET /telemetry polled once a second; history comes from the GET /events record stream. The stream is bounded and drops its oldest records under backpressure rather than stalling the engine, so the poll is authoritative for current state.',
  },
} as const satisfies Record<string, Definition>

export type GlossaryKey = keyof typeof GLOSSARY
