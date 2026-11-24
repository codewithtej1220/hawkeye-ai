import React, { useCallback, useEffect, useState } from 'react';
import { CheckCircle, ChevronLeft, ChevronRight, Download, RefreshCw, Search, ShieldAlert, X } from 'lucide-react';
import { toast } from 'sonner';
import {
  LEVEL_STYLES,
  LogEvent,
  LogStats,
  THREAT_LEVELS,
  ThreatLevel,
  cardClass,
  formatDateTime,
  formatMetres,
  getApiBase,
} from './shared';

const PAGE_SIZE = 25;

const EVENT_TYPE_TEXT: Record<string, string> = {
  level_change: 'Level change',
  update: 'Ongoing',
  analysis: 'Media analysis',
};

const RANGE_OPTIONS: { id: string; label: string; hours: number | null }[] = [
  { id: 'all', label: 'All time', hours: null },
  { id: '1h', label: 'Last hour', hours: 1 },
  { id: '24h', label: 'Last 24 hours', hours: 24 },
  { id: '7d', label: 'Last 7 days', hours: 24 * 7 },
];

interface EventLogProps {
  /** Newest event pushed over the live connection; triggers a refresh. */
  latestEvent: LogEvent | null;
}

const inputClass =
  'rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:focus:ring-blue-950';

export default function EventLog({ latestEvent }: EventLogProps) {
  const [events, setEvents] = useState<LogEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<LogStats | null>(null);
  const [page, setPage] = useState(0);
  const [level, setLevel] = useState<ThreatLevel | 'All' | 'Threats'>('All');
  const [source, setSource] = useState<'all' | 'live' | 'upload'>('all');
  const [range, setRange] = useState('all');
  const [search, setSearch] = useState('');
  const [onlyOpen, setOnlyOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<LogEvent | null>(null);

  const buildQuery = useCallback(() => {
    const params = new URLSearchParams();
    if (level === 'Threats') {
      params.set('min_level', 'High');
    } else if (level !== 'All') {
      params.set('level', level);
    }
    if (source !== 'all') {
      params.set('source', source);
    }
    const hours = RANGE_OPTIONS.find((r) => r.id === range)?.hours;
    if (hours) {
      params.set('since', new Date(Date.now() - hours * 3600 * 1000).toISOString());
    }
    if (search.trim()) {
      params.set('search', search.trim());
    }
    if (onlyOpen) {
      params.set('unacknowledged', 'true');
    }
    return params;
  }, [level, source, range, search, onlyOpen]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = buildQuery();
      params.set('limit', String(PAGE_SIZE));
      params.set('offset', String(page * PAGE_SIZE));
      const [listResponse, statsResponse] = await Promise.all([
        fetch(`${getApiBase()}/logs?${params.toString()}`),
        fetch(`${getApiBase()}/logs/stats`),
      ]);
      if (!listResponse.ok || !statsResponse.ok) {
        throw new Error(`Backend returned ${listResponse.status}`);
      }
      const list = (await listResponse.json()) as { total: number; events: LogEvent[] };
      setEvents(list.events);
      setTotal(list.total);
      setStats((await statsResponse.json()) as LogStats);
      setError(null);
    } catch (err) {
      setError('Could not load the event log. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, [buildQuery, page]);

  useEffect(() => {
    void load();
  }, [load, latestEvent?.id]);

  // Any filter change starts again from the first page
  useEffect(() => {
    setPage(0);
  }, [level, source, range, search, onlyOpen]);

  const openEvent = async (event: LogEvent) => {
    setSelected(event);
    try {
      const response = await fetch(`${getApiBase()}/logs/${event.id}`);
      if (response.ok) {
        setSelected((await response.json()) as LogEvent);
      }
    } catch {
      /* the summary row is already shown */
    }
  };

  const exportUrl = `${getApiBase()}/logs/export.csv?${buildQuery().toString()}`;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="flex flex-col gap-5">
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
        <SummaryTile label="Events recorded" value={stats?.total ?? 0} />
        {THREAT_LEVELS.map((name) => (
          <SummaryTile key={name} label={`${name} (24 h: ${stats?.last_24h?.[name] ?? 0})`} value={stats?.by_level?.[name] ?? 0} level={name} />
        ))}
        <SummaryTile label="Threats awaiting review" value={stats?.unacknowledged_threats ?? 0} alert={(stats?.unacknowledged_threats ?? 0) > 0} />
      </section>

      <section className={cardClass}>
        <div className="flex flex-col gap-3 border-b border-slate-200 p-4 dark:border-slate-800">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-950 dark:text-white">Event log</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Every threat-level change, ongoing threat and media analysis, kept on the server with a snapshot. {total} matching.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={() => void load()} className={`${inputClass} inline-flex items-center gap-2 font-medium`}>
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              <a href={exportUrl} className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-200 dark:focus:ring-blue-900">
                <Download className="h-4 w-4" />
                Export CSV
              </a>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative min-w-[220px] flex-1">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                className={`${inputClass} w-full pl-9`}
                placeholder="Search descriptions and review notes"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <select className={inputClass} value={level} onChange={(e) => setLevel(e.target.value as ThreatLevel | 'All' | 'Threats')} aria-label="Threat level">
              <option value="All">All levels</option>
              <option value="Threats">High and Critical</option>
              {THREAT_LEVELS.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <select className={inputClass} value={source} onChange={(e) => setSource(e.target.value as 'all' | 'live' | 'upload')} aria-label="Source">
              <option value="all">All sources</option>
              <option value="live">Live drone</option>
              <option value="upload">Uploaded media</option>
            </select>
            <select className={inputClass} value={range} onChange={(e) => setRange(e.target.value)} aria-label="Time range">
              {RANGE_OPTIONS.map((option) => (
                <option key={option.id} value={option.id}>{option.label}</option>
              ))}
            </select>
            <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
              <input type="checkbox" className="h-4 w-4 rounded border-slate-300" checked={onlyOpen} onChange={(e) => setOnlyOpen(e.target.checked)} />
              Not yet reviewed
            </label>
          </div>
        </div>

        <div className="overflow-x-auto">
          {error ? (
            <p className="px-6 py-12 text-center text-sm text-red-600 dark:text-red-400">{error}</p>
          ) : events.length === 0 ? (
            <div className="px-6 py-14 text-center text-slate-500 dark:text-slate-400">
              <p className="text-base font-medium text-slate-900 dark:text-white">No events match</p>
              <p className="mt-1 text-sm">Events are recorded when the threat level changes, while a threat continues, and for every analysed upload.</p>
            </div>
          ) : (
            <table className="w-full min-w-[980px] border-collapse text-left">
              <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-semibold">#</th>
                  <th className="px-4 py-3 font-semibold">Time</th>
                  <th className="px-4 py-3 font-semibold">Snapshot</th>
                  <th className="px-4 py-3 font-semibold">Level</th>
                  <th className="px-4 py-3 font-semibold">What happened</th>
                  <th className="px-4 py-3 font-semibold">Sensing</th>
                  <th className="px-4 py-3 font-semibold">Altitude / range</th>
                  <th className="px-4 py-3 text-right font-semibold">Review</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-sm dark:divide-slate-800">
                {events.map((event) => (
                  <tr key={event.id} className="cursor-pointer transition hover:bg-slate-50 dark:hover:bg-slate-800/60" onClick={() => void openEvent(event)}>
                    <td className="px-4 py-3 tabular-nums text-slate-500 dark:text-slate-400">{event.id}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-slate-700 dark:text-slate-200">{formatDateTime(event.ts)}</td>
                    <td className="px-4 py-3">
                      <div className="h-12 w-16 overflow-hidden rounded-lg border border-slate-200 bg-slate-900 dark:border-slate-700">
                        {event.has_snapshot && (
                          <img src={`${getApiBase()}/logs/${event.id}/snapshot`} alt="" loading="lazy" className="h-full w-full object-cover" />
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${LEVEL_STYLES[event.level].badge}`}>{event.level}</span>
                      <p className="mt-1.5 text-xs text-slate-500 dark:text-slate-400">
                        {EVENT_TYPE_TEXT[event.event_type] ?? event.event_type}
                        {event.event_type === 'level_change' && event.previous_level ? ` (was ${event.previous_level})` : ''}
                      </p>
                    </td>
                    <td className="max-w-[340px] px-4 py-3 text-slate-700 dark:text-slate-200">{event.description}</td>
                    <td className="px-4 py-3 text-slate-600 dark:text-slate-300">
                      {event.method_label ?? '-'}
                      {event.conditions && event.conditions !== 'normal' ? <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{event.conditions}</p> : null}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 tabular-nums text-slate-600 dark:text-slate-300">
                      {event.source === 'live' ? `${formatMetres(event.altitude_m)} / ${event.range_m ? `~${formatMetres(event.range_m)}` : '-'}` : 'upload'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {event.acknowledged ? (
                        <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300">
                          <CheckCircle className="h-3.5 w-3.5" />
                          Reviewed
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-xs font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300">
                          Open
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-slate-200 px-4 py-3 text-sm text-slate-600 dark:border-slate-800 dark:text-slate-300">
          <span>
            Page {page + 1} of {pages}
          </span>
          <div className="flex items-center gap-2">
            <button type="button" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))} className={`${inputClass} inline-flex items-center gap-1 disabled:opacity-40`}>
              <ChevronLeft className="h-4 w-4" />
              Newer
            </button>
            <button type="button" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)} className={`${inputClass} inline-flex items-center gap-1 disabled:opacity-40`}>
              Older
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </section>

      {selected && (
        <EventDetail
          event={selected}
          onClose={() => setSelected(null)}
          onUpdated={(updated) => {
            setSelected(updated);
            void load();
          }}
        />
      )}
    </div>
  );
}

function SummaryTile({ label, value, level, alert }: { label: string; value: number; level?: ThreatLevel; alert?: boolean }) {
  const style = level ? LEVEL_STYLES[level] : null;
  return (
    <div className={`rounded-2xl border px-4 py-3 ${style ? `${style.soft} ${style.border}` : alert ? 'border-orange-200 bg-orange-50 dark:border-orange-900 dark:bg-orange-950/30' : 'border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900'}`}>
      <p className={`text-2xl font-semibold tabular-nums ${style ? style.text : alert ? 'text-orange-700 dark:text-orange-300' : 'text-slate-950 dark:text-white'}`}>{value}</p>
      <p className="mt-0.5 text-xs text-slate-600 dark:text-slate-300">{label}</p>
    </div>
  );
}

function EventDetail({ event, onClose, onUpdated }: { event: LogEvent; onClose: () => void; onUpdated: (event: LogEvent) => void }) {
  const [by, setBy] = useState(() => window.localStorage.getItem('hawkeye-reviewer') ?? '');
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const style = LEVEL_STYLES[event.level];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const acknowledge = async () => {
    setSaving(true);
    try {
      window.localStorage.setItem('hawkeye-reviewer', by);
      const response = await fetch(`${getApiBase()}/logs/${event.id}/acknowledge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ by: by.trim() || 'operator', note }),
      });
      if (!response.ok) {
        throw new Error(String(response.status));
      }
      onUpdated((await response.json()) as LogEvent);
      toast.success(`Event #${event.id} marked as reviewed`);
    } catch {
      toast.error('Could not save the review. Is the backend running?');
    } finally {
      setSaving(false);
    }
  };

  const people = (event.detections ?? []).filter((d) => d.kind === 'person');
  const weapons = (event.detections ?? []).filter((d) => d.kind === 'weapon' && d.evidence !== 'classifier' && d.evidence !== 'specialist');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Event ${event.id}`}
        className="relative max-h-[92vh] w-full max-w-5xl overflow-y-auto rounded-3xl border border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-slate-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 p-5 dark:border-slate-800">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-xl font-semibold text-slate-950 dark:text-white">Event #{event.id}</h3>
              <span className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${style.badge}`}>{event.level}</span>
              <span className="rounded-full border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-600 dark:border-slate-700 dark:text-slate-300">
                {event.source === 'live' ? 'Live drone' : 'Uploaded media'} · {EVENT_TYPE_TEXT[event.event_type] ?? event.event_type}
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{formatDateTime(event.ts)}</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-slate-200 text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid gap-5 p-5 lg:grid-cols-[1.4fr_1fr]">
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 dark:border-slate-700">
            {event.has_snapshot ? (
              <img src={`${getApiBase()}/logs/${event.id}/snapshot`} alt="Snapshot with detections" className="h-auto w-full object-contain" />
            ) : (
              <div className="flex aspect-video items-center justify-center text-sm text-slate-400">No snapshot stored</div>
            )}
          </div>

          <div className="flex flex-col gap-4">
            <div className={`rounded-2xl border p-4 ${style.soft} ${style.border}`}>
              <p className="text-sm leading-6 text-slate-800 dark:text-slate-100">{event.description}</p>
              {event.previous_level && event.event_type === 'level_change' ? (
                <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">Changed from {event.previous_level} to {event.level}.</p>
              ) : null}
            </div>

            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
              <Fact label="People" value={String(event.people)} />
              <Fact label="Armed people" value={String(event.armed_people)} />
              <Fact label="Weapons" value={String(event.weapons)} />
              <Fact label="Possibly armed (distant)" value={String(event.possible_armed)} />
              <Fact label="Average confidence" value={`${event.average_confidence}%`} />
              <Fact label="Sensing method" value={event.method_label ?? '-'} />
              <Fact label="Conditions" value={event.conditions || '-'} />
              <Fact label="Drone altitude" value={formatMetres(event.altitude_m)} />
              <Fact label="Nearest person (est.)" value={event.range_m ? `~${formatMetres(event.range_m)}` : '-'} />
              <Fact label="Camera zoom" value={event.telemetry?.zoom ? `${event.telemetry.zoom}x` : '-'} />
            </dl>

            {(people.length > 0 || weapons.length > 0) && (
              <div className="rounded-2xl border border-slate-200 p-3 text-sm dark:border-slate-800">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Detections</p>
                <ul className="space-y-1.5 text-slate-700 dark:text-slate-200">
                  {people.map((d, i) => (
                    <li key={`p${i}`} className="flex justify-between gap-3">
                      <span>{d.armed ? (d.armed_possible ? 'Person, possible weapon' : 'Armed person') : 'Person'}{d.range_m ? `, about ${formatMetres(d.range_m)}` : ''}</span>
                      <span className="tabular-nums text-slate-500 dark:text-slate-400">{Math.round(d.confidence * 100)}%</span>
                    </li>
                  ))}
                  {weapons.map((d, i) => (
                    <li key={`w${i}`} className="flex justify-between gap-3">
                      <span>{d.concealed ? 'Concealed weapon' : 'Weapon'}{d.evidence === 'zoom' ? ' (magnified look)' : ''}</span>
                      <span className="tabular-nums text-slate-500 dark:text-slate-400">{Math.round(d.confidence * 100)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="rounded-2xl border border-slate-200 p-4 dark:border-slate-800">
              {event.acknowledged ? (
                <div className="flex items-start gap-3">
                  <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                  <div className="text-sm">
                    <p className="font-semibold text-slate-950 dark:text-white">
                      Reviewed by {event.ack_by}
                      {event.ack_at ? ` on ${formatDateTime(event.ack_at)}` : ''}
                    </p>
                    {event.ack_note ? <p className="mt-1 text-slate-600 dark:text-slate-300">{event.ack_note}</p> : null}
                  </div>
                </div>
              ) : (
                <div className="flex flex-col gap-2.5">
                  <p className="flex items-center gap-2 text-sm font-semibold text-slate-950 dark:text-white">
                    <ShieldAlert className="h-4 w-4" />
                    Mark as reviewed
                  </p>
                  <input className={inputClass} placeholder="Your name or badge" value={by} maxLength={80} onChange={(e) => setBy(e.target.value)} />
                  <textarea className={`${inputClass} min-h-[72px]`} placeholder="Note (what was done, outcome)" value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} />
                  <button
                    type="button"
                    disabled={saving}
                    onClick={() => void acknowledge()}
                    className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-200 disabled:opacity-60 dark:focus:ring-blue-900"
                  >
                    {saving ? 'Saving...' : 'Save review'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className="font-medium text-slate-900 dark:text-white">{value}</dd>
    </div>
  );
}
