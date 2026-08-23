import type { ReactNode } from 'react'
import type { GlossaryKey } from '../lib/glossary'
import { Info, Term } from './tooltip'

export const cx = (...classes: Array<string | false | null | undefined>) =>
  classes.filter(Boolean).join(' ')

export function Panel({
  title,
  hint,
  note,
  action,
  children,
  className,
}: {
  title: string
  hint?: GlossaryKey
  note?: ReactNode
  action?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={cx('panel', className)}>
      <header className="panel__header">
        <h2>
          {title}
          {hint ? <Info k={hint} /> : null}
        </h2>
        {action ?? (note ? <div className="panel__note">{note}</div> : null)}
      </header>
      {children}
    </section>
  )
}

export type Tone = 'neutral' | 'accent' | 'warning' | 'danger'

export function Stat({
  value,
  unit,
  label,
  hint,
  tone = 'neutral',
}: {
  value: string
  unit?: string
  label: string
  hint?: GlossaryKey
  tone?: Tone
}) {
  return (
    <div className={cx('stat', tone !== 'neutral' && `stat--${tone}`)}>
      <div className="stat__value">
        {value}
        {unit ? <span>{unit}</span> : null}
      </div>
      <div className="stat__label">{hint ? <Term k={hint}>{label}</Term> : label}</div>
    </div>
  )
}

export function Meter({ fraction, color = 'var(--accent)' }: { fraction: number; color?: string }) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0))
  return (
    <span
      className="meter"
      role="meter"
      aria-valuenow={Math.round(clamped * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <span style={{ width: `${clamped * 100}%`, background: color }} />
    </span>
  )
}

export function Pill({ children, tone = 'neutral' }: { children: ReactNode; tone?: Tone }) {
  const suffix =
    tone === 'accent' ? 'ok' : tone === 'warning' ? 'warn' : tone === 'danger' ? 'error' : ''
  return <span className={cx('pill', suffix && `pill--${suffix}`)}>{children}</span>
}

export function StatusDot({ tone = 'neutral' }: { tone?: Tone }) {
  const suffix =
    tone === 'accent' ? 'ok' : tone === 'warning' ? 'warn' : tone === 'danger' ? 'error' : ''
  return <span className={cx('status-dot', suffix && `status-dot--${suffix}`)} aria-hidden="true" />
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}
