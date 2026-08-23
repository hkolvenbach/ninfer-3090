import type { Band, Sample } from '../components/charts'
import type { ThroughputRecord } from './records'

/**
 * Converts throughput samples to time-accurate bands.
 *
 * The reporter suppresses intervals with no activity and folds the skipped time into the next
 * sample's `interval_seconds`, so a record's window is `[timestamp - interval, timestamp]` and
 * successive records are not evenly spaced. Deriving the left edge from the previous sample's
 * timestamp instead would draw an idle gap as if it were busy.
 */
export function bands(
  records: readonly ThroughputRecord[],
  pick: (record: ThroughputRecord) => number,
): Band[] {
  return records.map((record) => ({
    start: record.timestamp_unix_ms - record.interval_seconds * 1000,
    end: record.timestamp_unix_ms,
    value: Math.max(0, pick(record)),
  }))
}

/**
 * Converts throughput samples to instantaneous points.
 *
 * The scheduler counts on a throughput record are a snapshot taken when the report was emitted,
 * not an average over its interval, so they belong at the record's own timestamp. Using `bands`
 * for them would claim an occupancy held for the whole interval that was never observed.
 */
export function samples(
  records: readonly ThroughputRecord[],
  pick: (record: ThroughputRecord) => number,
): Sample[] {
  return records.map((record) => ({
    at: record.timestamp_unix_ms,
    value: Math.max(0, pick(record)),
  }))
}

/** Most recent sample, or null when the series is empty. */
export function latest<T>(items: readonly T[]): T | null {
  return items.length === 0 ? null : items[items.length - 1]!
}

/** Wall-clock span the series covers, in seconds. */
export function span(records: readonly ThroughputRecord[]): number {
  if (records.length === 0) return 0
  const first = records[0]!
  const last = records[records.length - 1]!
  return (last.timestamp_unix_ms - first.timestamp_unix_ms) / 1000 + first.interval_seconds
}

/**
 * Interval-weighted mean of a per-second rate. A plain average over samples would overweight the
 * short intervals, which are exactly the busy ones.
 */
export function weightedRate(
  records: readonly ThroughputRecord[],
  pick: (record: ThroughputRecord) => number,
): number {
  let weight = 0
  let total = 0
  for (const record of records) {
    weight += record.interval_seconds
    total += pick(record) * record.interval_seconds
  }
  return weight === 0 ? 0 : total / weight
}
