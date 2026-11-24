import React from 'react';
import { BackendDetection, Telemetry, formatMetres } from './shared';

/**
 * Detection overlay drawn by the browser over the live video, so the video can play smoothly while
 * detection runs at its own pace. Boxes are thin corner brackets with a margin around the object,
 * and each person gets one small label outside the box, so the person and the weapon stay visible.
 */
export function DetectionOverlay({ detections, showLabels, compact = false }: {
  detections: BackendDetection[];
  showLabels: boolean;
  compact?: boolean;
}) {
  const people = new Map(detections.filter((d) => d.kind === 'person').map((d) => [d.id, d]));
  const items = detections.filter((d) => !(d.kind === 'weapon' && d.evidence === 'classifier'));
  return (
    <div className="pointer-events-none absolute inset-0">
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        {items.map((d) => {
          const armed = d.kind === 'weapon' || d.armed;
          const color = armed ? '#ef4444' : '#22c55e';
          const pad = Math.max(0.3, 0.03 * Math.max(d.bbox.width, d.bbox.height));
          const x1 = Math.max(0, d.bbox.x - pad);
          const y1 = Math.max(0, d.bbox.y - pad);
          const x2 = Math.min(100, d.bbox.x + d.bbox.width + pad);
          const y2 = Math.min(100, d.bbox.y + d.bbox.height + pad);
          const lx = Math.max(0.8, (x2 - x1) / 4);
          const ly = Math.max(0.8, (y2 - y1) / 4);
          const path = [
            `M${x1},${y1 + ly} V${y1} H${x1 + lx}`,
            `M${x2 - lx},${y1} H${x2} V${y1 + ly}`,
            `M${x1},${y2 - ly} V${y2} H${x1 + lx}`,
            `M${x2 - lx},${y2} H${x2} V${y2 - ly}`,
          ].join(' ');
          return (
            <path
              key={d.id}
              d={path}
              fill="none"
              stroke={color}
              strokeWidth={compact ? 1.5 : 2}
              vectorEffect="non-scaling-stroke"
              strokeLinecap="square"
              style={{ filter: 'drop-shadow(0 0 1px rgba(0,0,0,0.9))' }}
            />
          );
        })}
      </svg>
      {showLabels &&
        items
          .filter((d) => d.kind === 'person' || !people.has(d.associated_person_id ?? ''))
          .map((d) => {
            const pad = Math.max(0.3, 0.03 * Math.max(d.bbox.width, d.bbox.height));
            const top = d.bbox.y - pad;
            const above = top > 6;
            const weapon = d.kind === 'person'
              ? detections
                  .filter((w) => w.kind === 'weapon' && w.associated_person_id === d.id && w.evidence !== 'classifier')
                  .sort((a, b) => b.confidence - a.confidence)[0]
              : null;
            const armed = d.kind === 'weapon' || d.armed;
            const text = d.kind === 'weapon'
              ? `${d.concealed ? 'Concealed weapon' : 'Weapon'} ${Math.round(d.confidence * 100)}%`
              : `${armed ? 'Armed' : 'Person'} ${Math.round(d.confidence * 100)}%` +
                (weapon ? ` · ${weapon.concealed ? 'concealed ' : ''}weapon ${Math.round(weapon.confidence * 100)}%` : '') +
                (typeof d.range_m === 'number' ? ` · ${formatMetres(d.range_m)}` : '');
            return (
              <span
                key={`label-${d.id}`}
                className={`absolute whitespace-nowrap rounded px-1.5 py-0.5 font-medium text-white ${compact ? 'text-[9px]' : 'text-[11px]'}`}
                style={{
                  left: `${Math.min(Math.max(0, d.bbox.x - pad), 80)}%`,
                  top: above ? `${top}%` : `${Math.min(97, d.bbox.y + d.bbox.height + pad)}%`,
                  transform: above ? 'translateY(-110%)' : 'translateY(10%)',
                  background: 'rgba(15, 23, 42, 0.62)',
                  borderLeft: `3px solid ${armed ? '#ef4444' : '#22c55e'}`,
                }}
              >
                {text}
              </span>
            );
          })}
    </div>
  );
}

/**
 * Top-down map of the drone, its camera's field of view and (simulation ground truth) the soldier,
 * so a demo can always show where he is even when he is out of the camera's view.
 */
export function MiniMap({ telemetry }: { telemetry?: Telemetry | null }) {
  const size = 220;
  if (!telemetry || telemetry.north_m === undefined || telemetry.east_m === undefined) {
    return (
      <div className="flex h-[220px] items-center justify-center text-sm text-slate-500 dark:text-slate-400">
        Waiting for drone position...
      </div>
    );
  }
  const drone = { n: telemetry.north_m, e: telemetry.east_m ?? 0 };
  const target = telemetry.target_north_m !== undefined && telemetry.target_east_m !== undefined
    ? { n: telemetry.target_north_m, e: telemetry.target_east_m }
    : null;
  const patrol = telemetry.patrol_north_m !== undefined && telemetry.patrol_east_m !== undefined
    ? { n: telemetry.patrol_north_m, e: telemetry.patrol_east_m }
    : null;
  // centre between drone and soldier; scale so both fit with room for the field of view
  const centre = target ? { n: (drone.n + target.n) / 2, e: (drone.e + target.e) / 2 } : drone;
  const span = target ? Math.hypot(drone.n - target.n, drone.e - target.e) : 0;
  const metresAcross = Math.max(60, span * 1.6 + 30);
  const k = size / metresAcross;
  const toXY = (p: { n: number; e: number }) => ({ x: size / 2 + (p.e - centre.e) * k, y: size / 2 - (p.n - centre.n) * k });
  const d = toXY(drone);
  const heading = ((telemetry.heading_deg ?? 0) * Math.PI) / 180;
  const half = ((telemetry.camera_fov_deg ?? 90) * Math.PI) / 360;
  const reach = Math.max(span * k * 1.15, size * 0.35);
  const ray = (a: number) => ({ x: d.x + Math.sin(a) * reach, y: d.y - Math.cos(a) * reach });
  const left = ray(heading - half);
  const right = ray(heading + half);
  const t = target ? toXY(target) : null;
  const ringStep = metresAcross > 250 ? 100 : metresAcross > 120 ? 50 : 25;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="h-[220px] w-full" role="img" aria-label="Map of the drone and the soldier">
      <rect x="0" y="0" width={size} height={size} rx="14" className="fill-slate-100 dark:fill-slate-950" />
      {[1, 2, 3].map((i) => (
        <circle key={i} cx={d.x} cy={d.y} r={i * ringStep * k} fill="none" strokeDasharray="2 4"
          className="stroke-slate-300 dark:stroke-slate-700" strokeWidth="1" />
      ))}
      {patrol && (
        <rect x={toXY(patrol).x - 15 * k} y={toXY(patrol).y - 15 * k} width={30 * k} height={30 * k} rx="3"
          className="fill-emerald-500/10 stroke-emerald-500/50" strokeDasharray="3 3" />
      )}
      <path d={`M${d.x},${d.y} L${left.x},${left.y} A${reach},${reach} 0 0 1 ${right.x},${right.y} Z`}
        className="fill-sky-500/15 stroke-sky-500/40" strokeWidth="1" />
      {t && (
        <g>
          <line x1={d.x} y1={d.y} x2={t.x} y2={t.y} className="stroke-red-500/50" strokeWidth="1" strokeDasharray="3 3" />
          <circle cx={t.x} cy={t.y} r="9" className="fill-red-500/20" />
          <circle cx={t.x} cy={t.y} r="4.5" className="fill-red-500" />
        </g>
      )}
      <g transform={`translate(${d.x} ${d.y}) rotate(${telemetry.heading_deg ?? 0})`}>
        <path d="M0,-8 L6,7 L0,3 L-6,7 Z" className="fill-sky-500 stroke-white dark:stroke-slate-900" strokeWidth="1" />
      </g>
      <text x="10" y="18" className="fill-slate-500 text-[10px] dark:fill-slate-400">N ↑</text>
      <text x={size - 10} y={size - 10} textAnchor="end" className="fill-slate-500 text-[10px] dark:fill-slate-400">
        rings {ringStep} m
      </text>
    </svg>
  );
}
