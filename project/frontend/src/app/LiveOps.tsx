import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Camera,
  CheckCircle,
  Compass,
  Crosshair,
  Gauge,
  Layers,
  Map as MapIcon,
  Mountain,
  Pin,
  Radio,
  Ruler,
  ShieldAlert,
  Tag,
  Users,
  ZoomIn,
} from 'lucide-react';
import { DetectionOverlay, MiniMap } from './Overlay';
import {
  BackendDetection,
  LEVEL_STYLES,
  LivePayload,
  StreamInfo,
  VIEW_OPTIONS,
  ViewId,
  cardClass,
  describeSituation,
  detectionTitle,
  formatClock,
  formatMetres,
} from './shared';

interface LiveOpsProps {
  frame: LivePayload | null;
  stream: StreamInfo | null;
  connected: boolean;
}

const HISTORY_SIZE = 8; // recent captured frames kept for review

const EVIDENCE_TEXT: Record<string, string> = {
  detector: 'seen in the full frame',
  zoom: 'found on a magnified look at the person',
  classifier: 'judged from the person\'s outline (too small to box)',
  tracked: 'carried over from recent frames',
  specialist: 'seen by the thermal-rifle model in recent frames',
};

const SENSOR_TEXT: Record<string, string> = {
  rgb: 'colour camera',
  thermal: 'thermal camera',
  'rgb+thermal': 'both cameras',
};

export default function LiveOps({ frame, stream, connected }: LiveOpsProps) {
  // null = follow whichever method the system is using for the current conditions
  const [pinnedView, setPinnedView] = useState<ViewId | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  // The dashboard shows analysed (captured) frames, each with its own boxes, so boxes always sit on
  // the objects. Recent frames are kept so an operator can step back and look at one.
  const [history, setHistory] = useState<LivePayload[]>([]);
  const [inspectId, setInspectId] = useState<number | null>(null);

  useEffect(() => {
    if (frame) {
      setHistory((previous) =>
        previous[0]?.frame_id === frame.frame_id ? previous : [frame, ...previous].slice(0, HISTORY_SIZE));
    }
  }, [frame]);

  const live = Boolean(frame && stream?.live);
  const summary = frame?.summary;
  const level = summary?.level ?? 'Clear';
  const levelStyle = LEVEL_STYLES[level];
  const inspected = inspectId === null ? null : history.find((h) => h.frame_id === inspectId) ?? null;
  const shown = inspected ?? frame; // frame in the main view
  const views: Partial<Record<ViewId, string>> = shown?.views ?? {};
  const activeMethod = (shown?.method ?? null) as ViewId | null;
  const shownView: ViewId | null = pinnedView && views[pinnedView] ? pinnedView : activeMethod && views[activeMethod] ? activeMethod : null;
  const mainImage = shownView ? views[shownView] : shown?.image;
  const detections = shown?.detections ?? [];

  const people = useMemo(() => (frame?.detections ?? []).filter((d) => d.kind === 'person'), [frame]);
  // The backend only reports weapons it saw, or weaker evidence confirmed over several frames
  const weapons = useMemo(
    () => (frame?.detections ?? []).filter((d) => d.kind === 'weapon' && d.evidence !== 'classifier'),
    [frame],
  );
  const nearest = useMemo(() => {
    const ranges = people.map((p) => p.range_m).filter((r): r is number => typeof r === 'number');
    return ranges.length ? Math.min(...ranges) : null;
  }, [people]);

  const brief = !frame
    ? 'Waiting for the drone. Start the capture (RUN_ALL_HAWKEYE.bat or thermal.py) and frames appear here automatically.'
    : !live
    ? 'The drone stream has stopped. Showing the last frame received.'
    : summary && summary.total_people === 0 && summary.weapon_count === 0
    ? 'No people or weapons in view. Continue routine surveillance.'
    : describeSituation(summary!.total_people, summary!.armed_people, summary!.weapon_count, summary!.possible_armed ?? 0);
  const actionRequired = Boolean(summary && (summary.armed_people > 0 || summary.weapon_count > 0));
  const telemetry = frame?.telemetry;
  const env = frame?.environment;

  return (
    <div className="flex flex-col gap-5">
      {/* Threat banner */}
      <section className={`rounded-3xl border p-5 ${levelStyle.soft} ${levelStyle.border}`}>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className={`grid h-14 w-14 shrink-0 place-items-center rounded-2xl border bg-white/70 dark:bg-slate-950/40 ${levelStyle.border} ${levelStyle.text}`}>
              {actionRequired ? <ShieldAlert className="h-7 w-7" /> : <CheckCircle className="h-7 w-7" />}
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Current threat level</p>
              <p className={`text-3xl font-semibold leading-tight ${levelStyle.text}`}>{level}</p>
            </div>
          </div>
          <p className="max-w-2xl flex-1 text-sm leading-6 text-slate-700 dark:text-slate-200">{brief}</p>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full border px-3 py-1.5 text-sm font-semibold ${levelStyle.badge}`}>
              {actionRequired ? 'Action required' : 'Monitored'}
            </span>
            {frame?.event_id ? (
              <span className="rounded-full border border-slate-300 bg-white/70 px-3 py-1.5 text-xs font-medium text-slate-600 dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-300">
                Logged as event #{frame.event_id}
              </span>
            ) : null}
          </div>
        </div>
      </section>

      {/* Counters + drone telemetry */}
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">
        <Stat icon={<Users className="h-4 w-4" />} label="People" value={summary?.total_people ?? 0} />
        <Stat icon={<ShieldAlert className="h-4 w-4" />} label="Armed people" value={summary?.armed_people ?? 0} tone={summary && summary.armed_people > 0 ? 'alert' : undefined} />
        <Stat icon={<AlertTriangle className="h-4 w-4" />} label="Weapons" value={summary?.weapon_count ?? 0} tone={summary && summary.weapon_count > 0 ? 'alert' : undefined} />
        <Stat icon={<Ruler className="h-4 w-4" />} label="Nearest person" value={nearest === null ? '-' : `~${formatMetres(nearest)}`} hint="Estimated from apparent size" />
        <Stat icon={<Mountain className="h-4 w-4" />} label="Drone altitude" value={formatMetres(telemetry?.altitude_m)} hint={telemetry?.altitude_reference ?? 'Above the take-off point'} />
        <Stat icon={<Compass className="h-4 w-4" />} label="Heading / tilt" value={telemetry?.heading_deg === undefined ? '-' : `${telemetry.heading_deg}° / ${telemetry.camera_pitch_deg ?? 0}°`} hint="Compass heading / camera tilt" />
        <Stat icon={<ZoomIn className="h-4 w-4" />} label="Zoom" value={telemetry?.zoom ? `${telemetry.zoom}x` : '-'} hint={telemetry?.camera_fov_deg ? `Field of view ${telemetry.camera_fov_deg}°` : undefined} />
        <Stat
          icon={<Gauge className="h-4 w-4" />}
          label="Frames analysed"
          value={live ? `${stream?.fps ?? 0} / s` : connected ? 'idle' : 'offline'}
          hint={frame?.processing_ms ? `Each frame takes ${frame.processing_ms} ms to analyse` : undefined}
        />
      </section>

      {/* Main view + detections */}
      <section className="grid gap-5 xl:grid-cols-[1.55fr_1fr]">
        <div className={cardClass}>
          <div className="flex flex-col gap-3 border-b border-slate-200 p-4 md:flex-row md:items-center md:justify-between dark:border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-slate-950 dark:text-white">{inspected ? 'Captured frame' : 'Live view'}</h2>
                {inspected ? (
                  <button
                    type="button"
                    onClick={() => setInspectId(null)}
                    className="rounded-full bg-blue-600 px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-white hover:bg-blue-700"
                  >
                    Back to live
                  </button>
                ) : (
                  <LiveBadge live={live} waiting={!frame} />
                )}
              </div>
              <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">
                {shownView ? VIEW_OPTIONS.find((v) => v.id === shownView)?.label : 'Drone camera'}
                {pinnedView ? ' (pinned)' : frame?.method_label ? ' (chosen automatically for the conditions)' : ''}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={() => setShowLabels((v) => !v)}
                aria-pressed={showLabels}
                className="mr-1 inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-blue-300 dark:border-slate-700 dark:text-slate-200 dark:hover:border-blue-500"
                title="Show or hide the text labels (the outlines stay)"
              >
                <Tag className="h-3.5 w-3.5" />
                {showLabels ? 'Labels on' : 'Labels off'}
              </button>
              <ViewButton active={pinnedView === null} onClick={() => setPinnedView(null)} label="Auto" />
              {VIEW_OPTIONS.map((view) => (
                <ViewButton
                  key={view.id}
                  active={pinnedView === view.id}
                  inUse={activeMethod === view.id}
                  disabled={!views[view.id]}
                  onClick={() => setPinnedView(view.id)}
                  label={view.short}
                />
              ))}
            </div>
          </div>
          <div className="relative aspect-[4/3] w-full overflow-hidden rounded-b-3xl bg-slate-950">
            {mainImage ? (
              <>
                <img src={mainImage} alt="Captured drone frame" className={`h-full w-full object-contain ${live || inspected ? '' : 'opacity-60'}`} />
                <DetectionOverlay detections={detections} showLabels={showLabels} />
              </>
            ) : (
              <Placeholder />
            )}
            {shown && (activeMethod || shownView) && (
              <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2 text-xs font-semibold">
                <span className="rounded-md bg-slate-950/70 px-2 py-1 text-emerald-300">
                  Detector vision: {VIEW_OPTIONS.find((v) => v.id === activeMethod)?.label ?? 'unknown'}
                </span>
                {shownView && shownView !== activeMethod && (
                  <span className="rounded-md bg-slate-950/70 px-2 py-1 text-slate-100">
                    Showing: {VIEW_OPTIONS.find((v) => v.id === shownView)?.label}
                  </span>
                )}
              </div>
            )}
            {shown && (
              <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-2 text-xs font-medium">
                <span className="rounded-md bg-slate-950/60 px-2 py-1 text-slate-100">Frame {shown.frame_id}</span>
                <span className="rounded-md bg-slate-950/60 px-2 py-1 text-slate-100">{formatClock(shown.timestamp)}</span>
                <span className={`rounded-md bg-slate-950/60 px-2 py-1 ${LEVEL_STYLES[shown.summary.level].text}`}>{shown.summary.level}</span>
                {shown.telemetry?.target_action ? (
                  <span className="rounded-md bg-slate-950/60 px-2 py-1 text-slate-100">Sim soldier: {shown.telemetry.target_action}</span>
                ) : null}
              </div>
            )}
          </div>
          {/* Recent captured frames: click one to look at it, "Back to live" returns */}
          {history.length > 1 && (
            <div className="flex gap-2 overflow-x-auto border-t border-slate-200 p-3 dark:border-slate-800">
              {history.map((h) => {
                const image = (h.method && h.views?.[h.method as ViewId]) || Object.values(h.views ?? {})[0] || h.image;
                const selected = inspected ? inspected.frame_id === h.frame_id : h.frame_id === frame?.frame_id;
                return (
                  <button
                    key={h.frame_id}
                    type="button"
                    onClick={() => setInspectId(h.frame_id === frame?.frame_id ? null : h.frame_id)}
                    className={`relative aspect-[4/3] w-28 shrink-0 overflow-hidden rounded-lg border-2 bg-slate-950 transition ${
                      selected ? 'border-blue-500' : 'border-transparent hover:border-slate-400 dark:hover:border-slate-600'
                    }`}
                    title={`Frame ${h.frame_id}, ${formatClock(h.timestamp)}: ${h.summary.level}`}
                  >
                    {image && <img src={image} alt="" className="h-full w-full object-contain" />}
                    <DetectionOverlay detections={h.detections} showLabels={false} compact />
                    <span className={`absolute right-1 top-1 h-2 w-2 rounded-full ${LEVEL_STYLES[h.summary.level].bar}`} />
                    <span className="absolute bottom-0.5 left-1 text-[9px] font-medium text-white drop-shadow">{formatClock(h.timestamp)}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-5">
          <div className={`${cardClass} flex min-h-0 flex-1 flex-col`}>
            <div className="border-b border-slate-200 p-4 dark:border-slate-800">
              <h2 className="text-lg font-semibold text-slate-950 dark:text-white">Detections</h2>
              <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">
                {people.length} {people.length === 1 ? 'person' : 'people'}, {weapons.length} {weapons.length === 1 ? 'weapon' : 'weapons'} in this frame
              </p>
            </div>
            <div className="max-h-[330px] flex-1 space-y-2 overflow-y-auto p-4">
              {people.length === 0 && weapons.length === 0 ? (
                <p className="py-8 text-center text-sm text-slate-500 dark:text-slate-400">
                  {frame ? 'Nothing detected in this frame.' : 'No frames yet.'}
                </p>
              ) : (
                <>
                  {people.map((person, index) => (
                    <DetectionRow
                      key={person.id}
                      detection={person}
                      index={index + 1}
                      weapons={(frame?.detections ?? []).filter((w) => w.kind === 'weapon' && w.associated_person_id === person.id)}
                    />
                  ))}
                  {weapons.filter((w) => !w.associated_person_id).map((weapon) => (
                    <DetectionRow key={weapon.id} detection={weapon} weapons={[]} />
                  ))}
                </>
              )}
            </div>
          </div>

          <div className={`${cardClass} p-4`}>
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <MapIcon className="h-4 w-4 text-slate-500 dark:text-slate-400" />
                <h2 className="text-base font-semibold text-slate-950 dark:text-white">Map</h2>
              </div>
              {typeof telemetry?.target_range_m === 'number' && (
                <span className="text-xs text-slate-500 dark:text-slate-400">Soldier {formatMetres(telemetry.target_range_m)} from the drone</span>
              )}
            </div>
            <MiniMap telemetry={telemetry} />
            <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
              <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-sky-500" />Drone and camera view</span>
              <span><span className="mr-1 inline-block h-2 w-2 rounded-full bg-red-500" />Soldier (simulation ground truth)</span>
              <span><span className="mr-1 inline-block h-2 w-2 rounded-sm border border-emerald-500" />Patrol area</span>
            </p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Out of view? Press G in the simulation window to turn the drone onto him.</p>
          </div>

          <div className={`${cardClass} p-4`}>
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-slate-500 dark:text-slate-400" />
              <h2 className="text-base font-semibold text-slate-950 dark:text-white">Sensing</h2>
            </div>
            <p className="mt-2 text-sm text-slate-700 dark:text-slate-200">
              <span className="font-semibold">{frame?.method_label ?? 'No method yet'}</span>
              {frame?.method_reasons?.length ? `: ${frame.method_reasons.join('; ')}.` : ''}
            </p>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <Reliability label="Colour camera" value={env?.rgb_reliability} />
              <Reliability label="Thermal camera" value={env?.thermal_reliability} />
            </div>
            {env?.conditions?.length ? (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {env.conditions.map((condition) => (
                  <span key={condition} className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
                    {condition}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </section>

      {/* The same frame through every camera mode */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {VIEW_OPTIONS.map((view) => {
          const image = views[view.id];
          const inUse = activeMethod === view.id;
          return (
            <button
              key={view.id}
              type="button"
              disabled={!image}
              onClick={() => setPinnedView(pinnedView === view.id ? null : view.id)}
              className={`${cardClass} group overflow-hidden text-left transition focus:outline-none focus-visible:ring-4 focus-visible:ring-blue-200 disabled:cursor-default dark:focus-visible:ring-blue-900 ${
                shownView === view.id ? 'ring-2 ring-blue-500 dark:ring-blue-400' : image ? 'hover:border-blue-300 dark:hover:border-blue-600' : ''
              }`}
              title={image ? `Show ${view.label} in the main view` : 'This camera is not sending frames'}
            >
              <div className="flex items-center justify-between gap-2 px-4 py-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-950 dark:text-white">{view.label}</p>
                  <p className="truncate text-xs text-slate-500 dark:text-slate-400">{view.sensor}</p>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  {pinnedView === view.id && <Pin className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />}
                  {inUse && (
                    <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300">
                      In use
                    </span>
                  )}
                </div>
              </div>
              <div className="relative aspect-[4/3] w-full bg-slate-950">
                {image ? (
                  <>
                    <img src={image} alt={`${view.label} view`} className={`h-full w-full object-contain ${live ? '' : 'opacity-60'}`} />
                    <DetectionOverlay detections={detections} showLabels={false} compact />
                  </>
                ) : (
                  <div className="flex h-full items-center justify-center text-xs text-slate-500">No signal</div>
                )}
              </div>
            </button>
          );
        })}
      </section>
    </div>
  );
}

function Stat({ icon, label, value, hint, tone }: { icon: React.ReactNode; label: string; value: React.ReactNode; hint?: string; tone?: 'alert' }) {
  return (
    <div className={`${cardClass} px-4 py-3`} title={hint}>
      <div className="flex items-center gap-2 text-xs font-medium text-slate-500 dark:text-slate-400">
        {icon}
        <span className="truncate">{label}</span>
      </div>
      <p className={`mt-1 truncate text-2xl font-semibold tabular-nums ${tone === 'alert' ? 'text-orange-600 dark:text-orange-400' : 'text-slate-950 dark:text-white'}`}>
        {value}
      </p>
    </div>
  );
}

function LiveBadge({ live, waiting }: { live: boolean; waiting: boolean }) {
  if (live) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-red-600 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-white">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" />
        Live
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-200 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 dark:bg-slate-800 dark:text-slate-300">
      <Radio className="h-3 w-3" />
      {waiting ? 'Waiting' : 'Stream stopped'}
    </span>
  );
}

function ViewButton({ active, inUse, disabled, onClick, label }: { active: boolean; inUse?: boolean; disabled?: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-pressed={active}
      title={inUse ? `The detector is analysing this view (${label})` : undefined}
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold transition focus:outline-none focus-visible:ring-4 focus-visible:ring-blue-200 disabled:cursor-not-allowed disabled:opacity-40 dark:focus-visible:ring-blue-900 ${
        active
          ? 'border-blue-600 bg-blue-600 text-white'
          : inUse
          ? 'border-emerald-500 bg-emerald-50 text-emerald-800 dark:border-emerald-500 dark:bg-emerald-950/50 dark:text-emerald-200'
          : 'border-slate-200 bg-white text-slate-700 hover:border-blue-300 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-blue-500'
      }`}
    >
      {inUse && <span aria-hidden className="h-2 w-2 rounded-full bg-emerald-500" />}
      {label}
      {inUse && <span className="sr-only"> (in use by the detector)</span>}
    </button>
  );
}

function DetectionRow({ detection, weapons, index }: { detection: BackendDetection; weapons: BackendDetection[]; index?: number }) {
  const threat = detection.kind === 'weapon' || detection.armed;
  const tone = threat
    ? 'border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/30'
    : 'border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-800/50';
  const facts: string[] = [];
  if (typeof detection.range_m === 'number') {
    facts.push(`about ${formatMetres(detection.range_m)} away`);
  }
  if (detection.sensor && SENSOR_TEXT[detection.sensor]) {
    facts.push(`seen by ${SENSOR_TEXT[detection.sensor]}`);
  }
  if (detection.armed_frames && detection.armed) {
    facts.push(`armed in ${detection.armed_frames} recent frames`);
  }
  return (
    <div className={`rounded-2xl border p-3 ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-2.5">
          <span className={`mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg ${threat ? 'bg-red-600 text-white' : 'bg-emerald-600 text-white'}`}>
            {detection.kind === 'weapon' ? <Crosshair className="h-4 w-4" /> : <Users className="h-4 w-4" />}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-slate-950 dark:text-white">
              {detectionTitle(detection)}
              {detection.kind === 'person' && index ? <span className="font-normal text-slate-500 dark:text-slate-400"> #{detection.track_id ?? index}</span> : null}
            </p>
            {facts.length > 0 && <p className="mt-0.5 text-xs text-slate-600 dark:text-slate-300">{facts.join(' · ')}</p>}
          </div>
        </div>
        <span className="shrink-0 rounded-full bg-white px-2 py-0.5 text-xs font-semibold tabular-nums text-slate-700 dark:bg-slate-900 dark:text-slate-200">
          {Math.round(detection.confidence * 100)}%
        </span>
      </div>
      {weapons.filter((w) => w.evidence !== 'classifier').map((weapon) => (
        <p key={weapon.id} className="mt-2 flex items-center gap-2 border-t border-red-200/70 pt-2 text-xs text-slate-700 dark:border-red-900/60 dark:text-slate-200">
          <Crosshair className="h-3.5 w-3.5 shrink-0 text-red-600 dark:text-red-400" />
          <span>
            <span className="font-semibold">{weapon.concealed ? 'Concealed weapon' : 'Weapon'} {Math.round(weapon.confidence * 100)}%</span>
            {weapon.evidence && EVIDENCE_TEXT[weapon.evidence] ? `: ${EVIDENCE_TEXT[weapon.evidence]}` : ''}
          </span>
        </p>
      ))}
    </div>
  );
}

function Reliability({ label, value }: { label: string; value?: number | null }) {
  const percent = typeof value === 'number' ? Math.round(value * 100) : null;
  const bar = percent === null ? 'bg-slate-300' : percent >= 60 ? 'bg-emerald-500' : percent >= 25 ? 'bg-amber-500' : 'bg-red-500';
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-300">
        <span>{label}</span>
        <span className="font-semibold tabular-nums">{percent === null ? 'no signal' : `${percent}% reliable`}</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
        <div className={`h-full rounded-full transition-all duration-500 ${bar}`} style={{ width: `${percent ?? 0}%` }} />
      </div>
    </div>
  );
}

function Placeholder() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-slate-400">
      <Camera className="h-10 w-10" />
      <p className="text-base font-medium text-slate-100">No live frames yet</p>
      <p className="max-w-sm text-sm">Start the drone capture and the live view, the four camera panels and the detections fill in automatically.</p>
    </div>
  );
}
