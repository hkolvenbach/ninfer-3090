// ECharts binding and the dashboard's shared chart theme.
//
// Only the pieces actually used are registered, so the bundle carries the bar/line/pie renderers
// and the tooltip/grid components rather than all of ECharts.

import { BarChart, CustomChart, LineChart, PieChart } from 'echarts/charts'
import {
  DatasetComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts/types/dist/shared'
import { useEffect, useRef } from 'react'
import { CHART } from '../lib/palette'

echarts.use([
  BarChart,
  CustomChart,
  LineChart,
  PieChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DatasetComponent,
  MarkLineComponent,
  MarkAreaComponent,
  CanvasRenderer,
])

export type { EChartsOption }
export { CHART }

const FONT =
  "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

/** Axis, grid, and tooltip defaults shared by every chart so they read as one instrument. */
export function baseOption(): EChartsOption {
  return {
    animation: false,
    textStyle: { fontFamily: FONT, fontSize: 10, color: CHART.muted },
    grid: { left: 44, right: 10, top: 12, bottom: 20, containLabel: false },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(8, 10, 11, 0.96)',
      borderColor: CHART.line,
      borderWidth: 1,
      padding: [7, 9],
      textStyle: { color: CHART.text, fontFamily: FONT, fontSize: 10 },
      axisPointer: { type: 'line', lineStyle: { color: CHART.dim, width: 1, type: 'dashed' } },
      extraCssText: 'border-radius:3px;box-shadow:0 6px 20px rgba(0,0,0,.55);max-width:320px;',
    },
    xAxis: {
      type: 'time',
      axisLine: { lineStyle: { color: CHART.line } },
      axisTick: { show: false },
      axisLabel: { color: CHART.dim, fontSize: 9, hideOverlap: true },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: CHART.dim, fontSize: 9 },
      splitLine: { lineStyle: { color: CHART.lineSoft, type: 'solid' } },
    },
  }
}

/**
 * Mounts one ECharts instance and applies `option` on change.
 *
 * `notMerge` is deliberate: series are rebuilt from state on every update, and merging would
 * retain series from a previous shape - for example after a replayed file replaces live data.
 */
export function EChart({ option, height = 132 }: { option: EChartsOption; height?: number }) {
  const host = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (host.current === null) return
    const instance = echarts.init(host.current, undefined, { renderer: 'canvas' })
    chart.current = instance
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(host.current)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [])

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true })
  }, [option])

  return <div ref={host} style={{ width: '100%', height }} />
}
