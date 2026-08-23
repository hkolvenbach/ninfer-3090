// Explanatory tooltips.
//
// The bubble renders in a portal at fixed coordinates rather than inside the trigger. Panels
// scroll and clip, and several triggers sit in the last row or last column of a grid, so an
// absolutely-positioned child would be cut off exactly where the explanation is needed.

import {
  useCallback,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { GLOSSARY, type GlossaryKey } from '../lib/glossary'
import { cx } from './ui'

const MARGIN = 8
const GAP = 9

interface Position {
  left: number
  top: number
  placement: 'top' | 'bottom'
}

function place(anchor: DOMRect, bubble: DOMRect): Position {
  const left = Math.min(
    Math.max(MARGIN, anchor.left + anchor.width / 2 - bubble.width / 2),
    window.innerWidth - bubble.width - MARGIN,
  )
  const above = anchor.top - GAP - bubble.height
  // Flip under the trigger only when there is genuinely no room above it.
  return above >= MARGIN
    ? { left, top: above, placement: 'top' }
    : { left, top: anchor.bottom + GAP, placement: 'bottom' }
}

/**
 * Wraps `children` in a hover/focus target that reveals `title` and `body`.
 *
 * Exposed as `aria-describedby` so the explanation is available to a screen reader and to
 * keyboard users, not only on pointer hover.
 */
export function Tooltip({
  title,
  body,
  children,
  className,
  style,
}: {
  title?: string
  body: ReactNode
  children: ReactNode
  className?: string
  /**
   * Applied to the trigger. The trigger is a real layout box - in a stacked bar it is the flex
   * child that carries the segment's width - so callers need to size it rather than an inner
   * element that would resolve its percentage against a collapsed parent.
   */
  style?: CSSProperties
}) {
  const id = useId()
  const anchor = useRef<HTMLSpanElement>(null)
  const bubble = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState<Position | null>(null)

  useLayoutEffect(() => {
    if (!open || anchor.current === null || bubble.current === null) {
      setPosition(null)
      return
    }
    setPosition(
      place(anchor.current.getBoundingClientRect(), bubble.current.getBoundingClientRect()),
    )
  }, [open])

  const hide = useCallback(() => setOpen(false), [])

  useLayoutEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') hide()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('scroll', hide, true)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('scroll', hide, true)
    }
  }, [open, hide])

  return (
    <>
      <span
        ref={anchor}
        className={cx('tip', className)}
        style={style}
        tabIndex={0}
        aria-describedby={open ? id : undefined}
        onPointerEnter={() => setOpen(true)}
        onPointerLeave={hide}
        onFocus={() => setOpen(true)}
        onBlur={hide}
      >
        {children}
      </span>
      {open
        ? createPortal(
            <div
              ref={bubble}
              id={id}
              role="tooltip"
              className={cx(
                'tip__bubble',
                position && `tip__bubble--${position.placement}`,
                position === null && 'tip__bubble--measuring',
              )}
              style={position ? { left: position.left, top: position.top } : undefined}
            >
              {title ? <strong>{title}</strong> : null}
              <span>{body}</span>
            </div>,
            document.body,
          )
        : null}
    </>
  )
}

/** Tooltip sourced from the glossary, so wording lives in exactly one place. */
export function Term({
  k,
  children,
  className,
}: {
  k: GlossaryKey
  children: ReactNode
  className?: string
}) {
  const entry = GLOSSARY[k]
  return (
    <Tooltip title={entry.title} body={entry.body} className={cx('tip--term', className)}>
      {children}
    </Tooltip>
  )
}

/** Standalone marker for headers and captions that have no natural word to underline. */
export function Info({ k }: { k: GlossaryKey }) {
  const entry = GLOSSARY[k]
  return (
    <Tooltip title={entry.title} body={entry.body} className="tip--info">
      <span aria-label={`About ${entry.title}`}>?</span>
    </Tooltip>
  )
}
