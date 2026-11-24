export type ThreatLevel = 'Clear' | 'Elevated' | 'High' | 'Critical';

export const THREAT_LEVELS: ThreatLevel[] = ['Clear', 'Elevated', 'High', 'Critical'];

export const LEVEL_STYLES: Record<ThreatLevel, {
  badge: string;
  soft: string;
  border: string;
  text: string;
  box: string;
  bar: string;
}> = {
  Clear: {
    badge: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-300 dark:border-emerald-800',
    soft: 'bg-emerald-50 dark:bg-emerald-950/30',
    border: 'border-emerald-200 dark:border-emerald-900',
    text: 'text-emerald-700 dark:text-emerald-300',
    box: 'border-emerald-500',
    bar: 'bg-emerald-500',
  },
  Elevated: {
    badge: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/50 dark:text-amber-300 dark:border-amber-800',
    soft: 'bg-amber-50 dark:bg-amber-950/30',
    border: 'border-amber-200 dark:border-amber-900',
    text: 'text-amber-700 dark:text-amber-300',
    box: 'border-amber-500',
    bar: 'bg-amber-500',
  },
  High: {
    badge: 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950/50 dark:text-orange-300 dark:border-orange-800',
    soft: 'bg-orange-50 dark:bg-orange-950/30',
    border: 'border-orange-200 dark:border-orange-900',
    text: 'text-orange-700 dark:text-orange-300',
    box: 'border-orange-500',
    bar: 'bg-orange-500',
  },
  Critical: {
    badge: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/50 dark:text-red-300 dark:border-red-800',
    soft: 'bg-red-50 dark:bg-red-950/30',
    border: 'border-red-200 dark:border-red-900',
    text: 'text-red-700 dark:text-red-300',
    box: 'border-red-500',
    bar: 'bg-red-600',
  },
};

export const cardClass =
  'rounded-3xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900 dark:shadow-none';

const backendHost = () => (typeof window === 'undefined' ? 'localhost' : window.location.hostname || 'localhost');
// Override with VITE_BACKEND_PORT (e.g. in frontend/.env.local) if port 8000 is used by another app
const BACKEND_PORT = import.meta.env.VITE_BACKEND_PORT || '8000';

export const getApiBase = () => {
  const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'https:' : 'http:';
  return `${protocol}//${backendHost()}:${BACKEND_PORT}`;
};

export const getWebSocketUrl = () => {
  const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${backendHost()}:${BACKEND_PORT}/stream`;
};

/** Plain-language operator brief for a threat summary. */
/** Operator brief that also covers people flagged as possibly armed (same wording as the event log). */
export const describeSituation = (people: number, armed: number, weapons: number, possible = 0): string => {
  const unarmed = Math.max(0, people - armed - possible);
  const parts: string[] = [];
  if (armed > 0) {
    const who = `${armed} armed ${armed === 1 ? 'person' : 'people'}`;
    parts.push(weapons > 0 ? `${who} with ${weapons} weapon${weapons === 1 ? '' : 's'}` : `${who} (weapon seen in recent frames)`);
  } else if (weapons > 0) {
    parts.push(`${weapons} weapon${weapons === 1 ? '' : 's'} detected with no person attached`);
  }
  if (possible > 0) {
    parts.push(`${possible} ${possible === 1 ? 'person' : 'people'} possibly armed (verify)`);
  }
  if (unarmed > 0) {
    parts.push(`${unarmed} unarmed ${unarmed === 1 ? 'person' : 'people'}`);
  }
  if (parts.length === 0) {
    return 'No people or weapons in view.';
  }
  const text = parts.join(', ');
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.${armed > 0 || weapons > 0 ? ' Immediate assessment required.' : ''}`;
};

export const describeThreat = (people: number, armed: number, weapons: number): string => {
  if (people === 0 && weapons === 0) {
    return 'No people or weapons detected.';
  }
  if (armed === 0 && weapons === 0) {
    return `${people} unarmed ${people === 1 ? 'person' : 'people'} detected. Monitor for suspicious activity.`;
  }
  if (armed === 0) {
    return `${weapons} weapon${weapons === 1 ? '' : 's'} detected with no person attached${people ? ` (${people} ${people === 1 ? 'person' : 'people'} in view)` : ''}. Verify the object.`;
  }
  return `${armed} armed ${armed === 1 ? 'person' : 'people'} with ${weapons} weapon${weapons === 1 ? '' : 's'} detected. Immediate assessment required.`;
};

export type MethodId = 'auto' | 'rgb' | 'thermal_grey' | 'thermal_inferno' | 'rgbt';

export const METHOD_OPTIONS: { id: MethodId; label: string; help: string }[] = [
  { id: 'auto', label: 'Auto', help: 'Measures light, fog, glare and thermal contrast, then uses the most reliable sensor(s).' },
  { id: 'rgb', label: 'RGB (visible)', help: 'Colour camera only. Best in good daylight; weak at night, in fog/smoke or glare.' },
  { id: 'thermal_grey', label: 'Thermal greyscale', help: 'Thermal image in white-hot greyscale. Works in darkness and smoke.' },
  { id: 'thermal_inferno', label: 'Thermal INFERNO', help: 'Thermal image in the INFERNO palette (preferred thermal view). Works in darkness and smoke.' },
  { id: 'rgbt', label: 'RGBT fusion', help: 'Combines visible and thermal: each detection is weighted by how reliable each sensor is right now.' },
];

export interface EnvironmentInfo {
  rgb_brightness: number | null;
  rgb_contrast: number | null;
  rgb_haze: number | null;
  rgb_glare: number | null;
  thermal_contrast: number | null;
  rgb_reliability: number | null;
  thermal_reliability: number | null;
  conditions: string[];
}

export interface SensingInfo {
  method?: string | null;
  method_label?: string | null;
  method_reasons?: string[];
  environment?: EnvironmentInfo | null;
}

// ---------------------------------------------------------------------------------------------
// Live stream + event log (backend/app/schemas.py, backend/app/eventlog.py)
// ---------------------------------------------------------------------------------------------

export interface BackendDetection {
  id: string;
  label: string;
  kind: 'person' | 'weapon' | 'other';
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  armed: boolean;
  weapon_count: number;
  associated_person_id?: string | null;
  sensor?: string | null;
  evidence?: string | null;
  concealed?: boolean;
  armed_evidence?: string | null;
  track_id?: number | null;
  armed_frames?: string | null;
  armed_confirmed?: boolean;
  armed_possible?: boolean;
  range_m?: number | null;
}

export interface BackendSummary {
  level: ThreatLevel;
  total_people: number;
  armed_people: number;
  weapon_count: number;
  risk_score: number;
  average_confidence: number;
  possible_armed?: number;
}

export interface Telemetry {
  altitude_m?: number;
  north_m?: number;
  east_m?: number;
  speed_mps?: number;
  heading_deg?: number;
  camera_pitch_deg?: number;
  camera_fov_deg?: number;
  zoom?: number;
  mode?: string;
  target_range_m?: number;
  target_action?: string;
  altitude_reference?: string; // what altitude_m is measured from, e.g. "Above ground"
  // simulation ground truth, for the map
  target_north_m?: number;
  target_east_m?: number;
  target_height_above_m?: number;
  patrol_north_m?: number;
  patrol_east_m?: number;
}

export interface StreamInfo {
  frames: number;
  skipped: number;
  fps: number;
  last_frame_age_s: number | null;
  live: boolean;
}

export type ViewId = 'rgb' | 'thermal_grey' | 'thermal_inferno' | 'rgbt';

export const VIEW_OPTIONS: { id: ViewId; label: string; short: string; sensor: string }[] = [
  { id: 'rgb', label: 'RGB (visible light)', short: 'RGB', sensor: 'Colour camera' },
  { id: 'thermal_grey', label: 'Thermal greyscale', short: 'Thermal grey', sensor: 'Thermal camera, white-hot' },
  { id: 'thermal_inferno', label: 'Thermal INFERNO', short: 'INFERNO', sensor: 'Thermal camera, INFERNO palette' },
  { id: 'rgbt', label: 'RGBT sensor fusion', short: 'RGBT fusion', sensor: 'Colour + thermal combined' },
];

export interface LivePayload {
  frame_id: number;
  timestamp: string;
  source: string;
  image: string;
  annotated_image?: string | null;
  detections: BackendDetection[];
  summary: BackendSummary;
  method?: ViewId | null;
  method_label?: string | null;
  method_reasons?: string[];
  sensors?: string[];
  environment?: EnvironmentInfo | null;
  views?: Partial<Record<ViewId, string>>;
  telemetry?: Telemetry | null;
  processing_ms?: number | null;
  event_id?: number | null;
  stream?: StreamInfo;
}

export interface LogEvent {
  id: number;
  ts: string;
  source: 'live' | 'upload' | string;
  event_type: 'level_change' | 'update' | 'analysis' | string;
  level: ThreatLevel;
  previous_level?: ThreatLevel | null;
  people: number;
  armed_people: number;
  weapons: number;
  possible_armed: number;
  average_confidence: number;
  method?: string | null;
  method_label?: string | null;
  conditions?: string | null;
  altitude_m?: number | null;
  range_m?: number | null;
  description: string;
  telemetry?: Telemetry | null;
  detections?: {
    kind: string; label: string; confidence: number; bbox: number[]; armed?: boolean; concealed?: boolean;
    armed_possible?: boolean; sensor?: string; evidence?: string; armed_evidence?: string; range_m?: number;
  }[];
  has_snapshot: boolean;
  acknowledged: boolean;
  ack_by?: string | null;
  ack_note?: string | null;
  ack_at?: string | null;
}

export interface LogStats {
  total: number;
  by_level: Record<ThreatLevel, number>;
  last_24h: Record<ThreatLevel, number>;
  unacknowledged_threats: number;
  first_event: string | null;
  last_event: string | null;
}

export const LEVEL_RANK: Record<ThreatLevel, number> = { Clear: 0, Elevated: 1, High: 2, Critical: 3 };

export const formatMetres = (value?: number | null) =>
  value === undefined || value === null ? '-' : `${value >= 100 ? Math.round(value) : value.toFixed(1)} m`;

export const formatClock = (iso: string | Date) =>
  new Date(iso).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });

export const formatDateTime = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** How a detection is worded for an operator. */
export const detectionTitle = (d: BackendDetection): string => {
  if (d.kind === 'weapon') {
    return d.concealed ? 'Concealed weapon' : 'Weapon';
  }
  if (d.kind !== 'person') {
    return d.label;
  }
  if (!d.armed) {
    return 'Person';
  }
  if (d.armed_possible) {
    return d.armed_evidence === 'specialist' ? 'Person: possible weapon' : 'Person: possible weapon (distant)';
  }
  return 'Armed person';
};
