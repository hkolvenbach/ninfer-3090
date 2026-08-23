import { Term, Tooltip } from '../components/tooltip'
import { Empty, Panel, Pill } from '../components/ui'
import { summarizeByAdapter } from '../lib/derive'
import { bytes, count, percent, seconds } from '../lib/format'
import type { AdapterInventory, RequestDoneRecord } from '../lib/records'

/**
 * LoRA adapters: what is resident, and what each one served.
 *
 * The two halves come from different places and are both needed. The inventory is reported by the
 * engine, so an adapter that has taken no traffic still appears. Usage is derived from completed
 * request records, so it survives replay. An adapter with no rows has been loaded and paid for in
 * VRAM without being used, which is exactly the case a traffic-only view would hide.
 */
export function AdaptersPanel({
  inventory,
  requests,
}: {
  inventory: AdapterInventory | undefined
  requests: RequestDoneRecord[]
}) {
  const usage = summarizeByAdapter(requests)
  const used = new Map(usage.map((entry) => [entry.name, entry]))
  const registered = inventory?.names ?? []
  // Residency can only be asserted when the engine actually reported its bank. Without an
  // inventory the correct statement is "unknown", not "not loaded".
  const known = inventory !== undefined

  // Union of what is loaded and what appears in the records: a replayed log may name an adapter
  // this server no longer registers, and a live server may register one with no traffic.
  const names = [
    '',
    ...registered,
    ...usage.map((entry) => entry.name).filter((name) => name !== '' && !registered.includes(name)),
  ]
  const rows = names.filter((name, index) => names.indexOf(name) === index)

  if (registered.length === 0 && usage.every((entry) => entry.name === '')) {
    return (
      <Panel title="Adapters" note="LoRA">
        <Empty>no adapters registered</Empty>
      </Panel>
    )
  }

  const perAdapter = inventory && inventory.count > 0 ? inventory.device_bytes / inventory.count : 0

  return (
    <Panel
      title="Adapters"
      className="panel--wide"
      note={
        inventory && inventory.count > 0 ? (
          <>
            rank {inventory.rank} ·{' '}
            <Tooltip
              title="Adapter bank"
              body={`One resident bank of ${inventory.count} adapter(s) at ${bytes(
                perAdapter,
              )} each. Committed at startup in its own device arena, before KV capacity is resolved.`}
              className="tip--term"
            >
              {bytes(inventory.device_bytes)} vram
            </Tooltip>
          </>
        ) : known ? (
          'none registered'
        ) : (
          // A schema-14 log and a pre-inventory engine both land here: usage is derivable, the
          // resident bank is not.
          'usage only · no inventory reported'
        )
      }
    >
      <table className="table">
        <thead>
          <tr>
            <th>adapter</th>
            <th className="numeric">reqs</th>
            <th className="numeric">gen</th>
            <th className="numeric">
              <Term k="ttft">ttft p50</Term>
            </th>
            <th className="numeric">
              <Term k="decodePerRequest">tok/s</Term>
            </th>
            <th className="numeric">
              <Term k="prefillAvoided">reused</Term>
            </th>
            <th className="numeric">
              <Term k="mtpAccept">mtp</Term>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((name) => {
            const entry = used.get(name)
            const resident = name === '' || registered.includes(name)
            const absent = known && !resident
            return (
              <tr key={name || '<base>'}>
                <td className="emphasis">
                  <Tooltip
                    title={name === '' ? 'Base model' : name}
                    body={
                      name === ''
                        ? 'Requests served by the base weights, with no adapter applied.'
                        : resident
                          ? `Served as model id "${
                              inventory?.model_ids?.[registered.indexOf(name)] ?? name
                            }". Resident in the bank from startup; there is no load or unload path.`
                          : known
                            ? 'Served requests in this window but is not registered on the engine now reporting.'
                            : 'Served requests in this window. This source reports no adapter inventory, so residency is unknown.'
                    }
                    className="tip--term"
                  >
                    {name === '' ? 'base' : name}
                  </Tooltip>
                  {absent ? (
                    <span className="adapters__flag">
                      <Pill tone="warning">not loaded</Pill>
                    </span>
                  ) : null}
                </td>
                <td className="numeric emphasis">{entry ? count(entry.summary.count) : '—'}</td>
                <td className="numeric">{entry ? count(entry.generatedTokens) : '—'}</td>
                <td className="numeric">{entry ? seconds(entry.summary.ttft.p50) : '—'}</td>
                <td className="numeric">
                  {entry && entry.summary.decodeTokensPerSecond.p50 > 0
                    ? entry.summary.decodeTokensPerSecond.p50.toFixed(0)
                    : '—'}
                </td>
                <td className="numeric">{entry ? percent(entry.summary.prefillAvoided) : '—'}</td>
                <td className="numeric">
                  {entry && entry.summary.speculative.drafted > 0
                    ? percent(entry.summary.speculative.acceptRate)
                    : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="panel__footnote">
        Rows are grouped by the resolved adapter, not by the requested model id — on the Anthropic
        route an unknown model silently falls back to the base weights. A registered adapter with no
        rows is resident and costing VRAM without serving traffic.
      </p>
    </Panel>
  )
}
