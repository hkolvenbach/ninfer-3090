// Display formatting. Every reading in the dashboard is a number a maintainer will compare
// against a log line or a metrics scrape, so these keep magnitudes explicit and never round a
// value into a different order.

export function bytes(value: number, digits = 1): string {
  if (!Number.isFinite(value) || value <= 0) return '0'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let scaled = value
  let unit = 0
  while (scaled >= 1000 && unit < units.length - 1) {
    scaled /= 1000
    unit += 1
  }
  return `${scaled.toFixed(unit === 0 ? 0 : digits)} ${units[unit]}`
}

export function count(value: number): string {
  if (!Number.isFinite(value)) return '0'
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (Math.abs(value) >= 1e4) return `${(value / 1e3).toFixed(1)}k`
  return Math.round(value).toLocaleString('en-US')
}

export function percent(fraction: number, digits = 0): string {
  if (!Number.isFinite(fraction)) return '0%'
  return `${(fraction * 100).toFixed(digits)}%`
}

export function seconds(value: number): string {
  if (!Number.isFinite(value)) return '0s'
  if (value >= 60) {
    const minutes = Math.floor(value / 60)
    return `${minutes}m ${Math.round(value - minutes * 60)}s`
  }
  if (value >= 10) return `${value.toFixed(1)}s`
  if (value >= 0.01) return `${value.toFixed(2)}s`
  return `${(value * 1000).toFixed(0)}ms`
}

export function duration(value: number): string {
  if (!Number.isFinite(value) || value < 0) return '0s'
  const hours = Math.floor(value / 3600)
  const minutes = Math.floor((value % 3600) / 60)
  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m ${Math.floor(value % 60)}s`
  return `${Math.floor(value)}s`
}

export function rate(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0'
  if (value >= 1000) return Math.round(value).toLocaleString('en-US')
  return value.toFixed(value >= 100 ? 0 : 1)
}

export function clock(unixMs: number): string {
  return new Date(unixMs).toLocaleTimeString('en-US', { hour12: false })
}
