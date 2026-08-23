// Chart primitives, rendered with ECharts.
//
// The throughput reporter only emits an interval that had activity, so samples are irregularly
// spaced and a sample's `interval_seconds` is the authority on the window it covers. Every series
// here is therefore plotted against real wall-clock time, never against sample index, which would
// silently compress an idle gap into a straight line.

import type {
  CustomSeriesRenderItemAPI,
  CustomSeriesRenderItemParams,
} from 'echarts/types/dist/shared'
import { useMemo, type ReactNode } from 'react'
import { clock } from '../lib/format'
import { baseOption, CHART, EChart, type EChartsOption } from './echart'
import { Tooltip } from './tooltip'
import { cx } from './ui'

export interface Band {
  /** Left edge of the sample's window, in ms since the epoch. */
  start: number
  /** Right edge of the sample's window. */
  end: number
  value: number
}

export interface Sample {
  at: number
  value: number
}

function Frame({
  caption,
  legend,
  children,
}: {
  caption?: ReactNode
  legend?: ReactNode
  children: ReactNode
}) {
  return (
    <figure className="chart">
      {children}
      {(caption || legend) && (
        <figcaption>
          {legend}
          {caption ? <span className="chart__caption">{caption}</span> : null}
        </figcaption>
      )}
    </figure>
  )
}

function rows(entries: Array<{ color: string; label: string; value: string }>): string {
  return entries
    .map(
      (entry) =>
        `<div class="tipline"><i style="background:${entry.color}"></i>` +
        `<span>${entry.label}</span><b>${entry.value}</b></div>`,
    )
    .join('')
}

/**
 * Interval-average series drawn as rectangles spanning each sample's own window.
 *
 * Bars are rendered by a custom series because their widths differ: a bar must cover exactly the
 * interval it averaged. A uniform `barWidth` would erase the distinction between a busy 5 s
 * interval and one that folded idle time forward, which is the main thing this chart is read for.
 *
 * The tooltip is driven by an invisible line series at each window's midpoint. Axis-triggered
 * tooltips need series data addressable by axis value, and pinning that to explicit midpoints
 * keeps the hover target stable regardless of how the rectangles were laid out.
 */
export function BandChart({
  label,
  series,
  report,
  domain,
  ceiling,
  unit,
  caption,
  legend,
  height,
}: {
  label: string
  series: Array<{ name: string; bands: Band[]; color: string; opacity?: number }>
  /**
   * Series named in the tooltip, defaulting to the drawn ones. Lets two charts that plot related
   * quantities on independent scales still report both readings for the interval under the cursor.
   */
  report?: Array<{ name: string; bands: Band[]; color: string }>
  /** Pins the time axis, so sibling charts of the same records stay aligned. */
  domain?: [number, number]
  ceiling?: number
  unit?: string
  caption?: ReactNode
  legend?: ReactNode
  height?: number
}) {
  const option = useMemo<EChartsOption>(() => {
    const base = baseOption()
    const reported = report ?? series
    const windows = series[0]?.bands ?? []
    return {
      ...base,
      aria: { enabled: true, label: { description: label } },
      xAxis: { ...base.xAxis, min: domain?.[0], max: domain?.[1] },
      yAxis: { ...base.yAxis, max: ceiling, min: 0 },
      tooltip: {
        ...base.tooltip,
        formatter: (params: unknown) => {
          const list = Array.isArray(params) ? params : [params]
          const first = list[0] as { dataIndex?: number } | undefined
          const index = first?.dataIndex ?? -1
          const window = windows[index]
          if (window === undefined) return ''
          const seconds = (window.end - window.start) / 1000
          const head =
            `<div class="tiphead">${clock(window.start)} → ${clock(window.end)}` +
            `<em>${seconds.toFixed(1)}s interval</em></div>`
          return (
            head +
            rows(
              reported.map((entry) => ({
                color: entry.color,
                label: entry.name,
                value: `${(entry.bands[index]?.value ?? 0).toFixed(1)}${unit ? ` ${unit}` : ''}`,
              })),
            )
          )
        },
      },
      series: [
        ...series.map((entry) => ({
          type: 'custom' as const,
          name: entry.name,
          silent: true,
          // Each datum is [start, end, value]. Without an explicit encode ECharts assumes
          // dimension 1 is the y value and scales the axis to an epoch timestamp.
          dimensions: ['start', 'end', 'value'],
          encode: { x: [0, 1], y: 2 },
          data: entry.bands.map((band) => [band.start, band.end, band.value]),
          renderItem: (_params: CustomSeriesRenderItemParams, api: CustomSeriesRenderItemAPI) => {
            const left = api.coord([api.value(0) as number, api.value(2) as number])
            const right = api.coord([api.value(1) as number, 0])
            return {
              type: 'rect' as const,
              shape: {
                x: left[0],
                y: left[1],
                // Sub-pixel intervals still need to be visible as a discrete measurement.
                width: Math.max(0.8, right[0] - left[0]),
                height: Math.max(0, right[1] - left[1]),
              },
              style: { fill: entry.color, opacity: entry.opacity ?? 1 },
            }
          },
        })),
        {
          type: 'line' as const,
          silent: true,
          symbolSize: 0,
          lineStyle: { opacity: 0 },
          tooltip: { show: true },
          data: windows.map((band, index) => [
            (band.start + band.end) / 2,
            Math.max(...series.map((entry) => entry.bands[index]?.value ?? 0)),
          ]),
        },
      ],
    }
  }, [series, report, domain, ceiling, unit, label])

  return (
    <Frame caption={caption} legend={legend}>
      <EChart option={option} height={height} />
    </Frame>
  )
}

/**
 * Instantaneous samples on a real time axis.
 *
 * Distinct from BandChart because the two describe different quantities. A throughput sample is
 * an average over its whole interval, so it is honest to fill that interval. The scheduler's
 * running/waiting counts are a snapshot taken at the moment the report was emitted, so filling
 * the interval would assert an occupancy that was never measured. These are drawn as points at
 * their own timestamps, connected faintly only to show ordering.
 */
export function SampleChart({
  label,
  series,
  ceiling,
  unit,
  caption,
  legend,
  height,
}: {
  label: string
  series: Array<{ name: string; samples: Sample[]; color: string }>
  ceiling?: number
  unit?: string
  caption?: ReactNode
  legend?: ReactNode
  height?: number
}) {
  const option = useMemo<EChartsOption>(() => {
    const base = baseOption()
    return {
      ...base,
      aria: { enabled: true, label: { description: label } },
      yAxis: { ...base.yAxis, max: ceiling, min: 0, minInterval: 1 },
      tooltip: {
        ...base.tooltip,
        formatter: (params: unknown) => {
          const list = (Array.isArray(params) ? params : [params]) as Array<{
            value: [number, number]
            seriesName: string
            color: string
          }>
          if (list.length === 0) return ''
          const head = `<div class="tiphead">${clock(list[0]!.value[0])}<em>snapshot</em></div>`
          return (
            head +
            rows(
              list.map((entry) => ({
                color: entry.color,
                label: entry.seriesName,
                value: `${entry.value[1]}${unit ? ` ${unit}` : ''}`,
              })),
            )
          )
        },
      },
      series: series.map((entry) => ({
        type: 'line' as const,
        name: entry.name,
        data: entry.samples.map((sample) => [sample.at, sample.value]),
        showSymbol: true,
        symbolSize: 4,
        itemStyle: { color: entry.color },
        lineStyle: { color: entry.color, width: 1, opacity: 0.45 },
        emphasis: { scale: 1.6 },
      })),
    }
  }, [series, ceiling, unit, label])

  return (
    <Frame caption={caption} legend={legend}>
      <EChart option={option} height={height} />
    </Frame>
  )
}

/** A single filled line series, for the 1 Hz board polls. */
export function LineChart({
  label,
  name,
  samples,
  color,
  ceiling,
  unit,
  caption,
  legend,
  marks,
  height,
}: {
  label: string
  name: string
  samples: Sample[]
  color: string
  ceiling?: number
  unit?: string
  caption?: ReactNode
  legend?: ReactNode
  /** Per-sample annotation, used to flag board throttling. Empty string means unflagged. */
  marks?: string[]
  height?: number
}) {
  const option = useMemo<EChartsOption>(() => {
    const base = baseOption()
    return {
      ...base,
      aria: { enabled: true, label: { description: label } },
      yAxis: { ...base.yAxis, max: ceiling, min: 0 },
      tooltip: {
        ...base.tooltip,
        formatter: (params: unknown) => {
          const list = (Array.isArray(params) ? params : [params]) as Array<{
            value: [number, number]
            dataIndex: number
          }>
          const point = list[0]
          if (point === undefined) return ''
          const mark = marks?.[point.dataIndex]
          return (
            `<div class="tiphead">${clock(point.value[0])}</div>` +
            rows([
              {
                color,
                label: name,
                value: `${point.value[1].toFixed(0)}${unit ? ` ${unit}` : ''}`,
              },
            ]) +
            (mark ? `<div class="tipnote">throttled: ${mark}</div>` : '')
          )
        },
      },
      series: [
        {
          type: 'line' as const,
          name,
          data: samples.map((sample) => [sample.at, sample.value]),
          showSymbol: false,
          smooth: false,
          lineStyle: { color, width: 1.4 },
          areaStyle: { color, opacity: 0.13 },
          // Throttled samples are marked in place rather than in a separate series so the
          // annotation cannot drift from the reading it explains.
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: CHART.danger, width: 1, opacity: 0.45, type: 'solid' },
            label: { show: false },
            data: (marks ?? [])
              .map((mark, index) => (mark ? { xAxis: samples[index]?.at } : null))
              .filter((entry): entry is { xAxis: number } => entry !== null),
          },
        },
      ],
    }
  }, [samples, color, ceiling, unit, name, label, marks])

  return (
    <Frame caption={caption} legend={legend}>
      <EChart option={option} height={height} />
    </Frame>
  )
}

export interface Segment {
  label: string
  value: number
  color: string
  /** Optional explanation shown alongside the share when the segment is hovered. */
  hint?: string
  /** Overrides the formatted share, for segments measured in something other than a count. */
  display?: string
}

/**
 * Proportional horizontal bar, used for phase splits and categorical breakdowns.
 *
 * Each segment carries its own tooltip: a bare colour band is unreadable without one, and the
 * smallest segments are usually the interesting ones.
 */
export function StackedBar({ segments, height = 8 }: { segments: Segment[]; height?: number }) {
  const total = segments.reduce((sum, segment) => sum + Math.max(0, segment.value), 0)
  if (total <= 0) return <div className="stacked-bar" style={{ height }} />
  return (
    <div className="stacked-bar" style={{ height }}>
      {segments.map((segment) =>
        segment.value <= 0 ? null : (
          <Tooltip
            key={segment.label}
            title={segment.label}
            body={
              <>
                {segment.display ?? `${((segment.value / total) * 100).toFixed(1)}% of total`}
                {segment.hint ? <div className="tipnote">{segment.hint}</div> : null}
              </>
            }
            className="stacked-bar__segment"
            // The width belongs on the flex child. Sizing an inner element instead resolves the
            // percentage against a parent that has no width of its own, collapsing the segment.
            style={{ width: `${(segment.value / total) * 100}%` }}
          >
            <span style={{ background: segment.color }} />
          </Tooltip>
        ),
      )}
    </div>
  )
}

export function Legend({
  items,
}: {
  items: Array<{ label: string; color: string; hint?: string }>
}) {
  return (
    <span className="legend">
      {items.map((item) =>
        item.hint ? (
          <Tooltip key={item.label} title={item.label} body={item.hint} className="legend__item">
            <i style={{ background: item.color }} />
            {item.label}
          </Tooltip>
        ) : (
          <span key={item.label} className={cx('legend__item')}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ),
      )}
    </span>
  )
}
