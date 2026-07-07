import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Bird,
  Camera,
  CheckCircle,
  Clock,
  Eye,
  Filter,
  Moon,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sun,
  Users,
} from 'lucide-react';
import { Toaster, toast } from 'sonner';

const APP_TITLE = 'HawkEye AI - Threat Classification Dashboard';

const IMAGES = [
  'https://images.unsplash.com/photo-1523438927869-e5819c560724?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx0aGVybWFsJTIwbmlnaHQlMjB2aXNpb24lMjBkcm9uZXxlbnwxfHx8fDE3Nzk5NDI3MTZ8MA&ixlib=rb-4.1.0&q=80&w=1080',
  'https://images.unsplash.com/photo-1532989029401-439615f3d4b4?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkcm9uZSUyMHBvaW50JTIwb2YlMjB2aWV3JTIwbmF0dXJlfGVufDF8fHx8MTc3OTk0MjcxOXww&ixlib=rb-4.1.0&q=80&w=1080',
  'https://images.unsplash.com/photo-1495553715023-a3c367bb8620?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHxhZXJpYWwlMjB2aWV3JTIwcGVvcGxlJTIwd2Fsa2luZ3xlbnwxfHx8fDE3Nzk5NDI3MjJ8MA&ixlib=rb-4.1.0&q=80&w=1080',
];

type ThreatLevel = 'Clear' | 'Elevated' | 'High' | 'Critical';
type ThemeMode = 'light' | 'dark';

interface DetectionBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  armed: boolean;
  weaponCount: number;
  label: string;
  kind: 'person' | 'weapon' | 'other';
}

interface FrameData {
  imageUrl: string;
  annotatedImageUrl?: string;  // Image with bounding boxes drawn by model
  location: string;
  detections: DetectionBox[];
  summary?: ThreatSummary;
  timestamp?: string;
  source?: string;
}

interface BackendDetection {
  id: string;
  label: string;
  kind: 'person' | 'weapon' | 'other';
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  armed: boolean;
  weapon_count: number;
  associated_person_id?: string | null;
}

interface FramePayload {
  frame_id: number;
  timestamp: string;
  source: string;
  image: string;
  annotated_image?: string;  // Image with bounding boxes
  detections: BackendDetection[];
  summary: BackendThreatSummary;
}

interface BackendThreatSummary {
  level: ThreatLevel;
  total_people: number;
  armed_people: number;
  weapon_count: number;
  risk_score: number;
  average_confidence: number;
}

interface ThreatSummary {
  level: ThreatLevel;
  totalPeople: number;
  armedPeople: number;
  weaponCount: number;
  riskScore: number;
  averageConfidence: number;
}

interface ArchiveEntry extends ThreatSummary {
  id: string;
  timestamp: Date;
  imageUrl: string;
  annotatedImageUrl?: string;  // Image with bounding boxes
  status: 'Action Required' | 'Monitored';
}

const MOCK_FRAMES: FrameData[] = [
  {
    imageUrl: IMAGES[0],
    location: 'North gate walkway',
    detections: [
      { id: 'd1', x: 45, y: 55, width: 8, height: 15, confidence: 92, armed: false, weaponCount: 0 },
    ],
  },
  {
    imageUrl: IMAGES[1],
    location: 'Service road',
    detections: [
      { id: 'd2', x: 30, y: 40, width: 5, height: 12, confidence: 88, armed: true, weaponCount: 1 },
      { id: 'd3', x: 60, y: 65, width: 6, height: 14, confidence: 95, armed: false, weaponCount: 0 },
    ],
  },
  {
    imageUrl: IMAGES[2],
    location: 'Visitor parking edge',
    detections: [
      { id: 'd4', x: 50, y: 45, width: 10, height: 20, confidence: 98, armed: true, weaponCount: 2 },
      { id: 'd5', x: 15, y: 20, width: 4, height: 8, confidence: 81, armed: true, weaponCount: 1 },
    ],
  },
];

const LEVEL_STYLES: Record<ThreatLevel, {
  badge: string;
  soft: string;
  border: string;
  text: string;
  box: string;
}> = {
  Clear: {
    badge: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/50 dark:text-emerald-300 dark:border-emerald-800',
    soft: 'bg-emerald-50 dark:bg-emerald-950/30',
    border: 'border-emerald-200 dark:border-emerald-900',
    text: 'text-emerald-700 dark:text-emerald-300',
    box: 'border-emerald-500',
  },
  Elevated: {
    badge: 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/50 dark:text-amber-300 dark:border-amber-800',
    soft: 'bg-amber-50 dark:bg-amber-950/30',
    border: 'border-amber-200 dark:border-amber-900',
    text: 'text-amber-700 dark:text-amber-300',
    box: 'border-amber-500',
  },
  High: {
    badge: 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950/50 dark:text-orange-300 dark:border-orange-800',
    soft: 'bg-orange-50 dark:bg-orange-950/30',
    border: 'border-orange-200 dark:border-orange-900',
    text: 'text-orange-700 dark:text-orange-300',
    box: 'border-orange-500',
  },
  Critical: {
    badge: 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/50 dark:text-red-300 dark:border-red-800',
    soft: 'bg-red-50 dark:bg-red-950/30',
    border: 'border-red-200 dark:border-red-900',
    text: 'text-red-700 dark:text-red-300',
    box: 'border-red-500',
  },
};

const cardClass = 'rounded-3xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900 dark:shadow-none';

const getThreatLevel = (armedPeople: number, weaponCount: number): ThreatLevel => {
  if (armedPeople === 0 || weaponCount === 0) {
    return 'Clear';
  }

  const riskScore = armedPeople * weaponCount;

  if (riskScore >= 4) {
    return 'Critical';
  }

  if (riskScore >= 2) {
    return 'High';
  }

  return 'Elevated';
};

const analyzeFrame = (frame: FrameData): ThreatSummary => {
  const totalPeople = frame.detections.length;
  const armedPeople = frame.detections.filter((detection) => detection.armed).length;
  const weaponCount = frame.detections.reduce((total, detection) => total + detection.weaponCount, 0);
  const riskScore = armedPeople > 0 && weaponCount > 0 ? armedPeople * weaponCount : 0;
  const averageConfidence = Math.round(
    frame.detections.reduce((total, detection) => total + detection.confidence, 0) / Math.max(totalPeople, 1),
  );

  return {
    level: getThreatLevel(armedPeople, weaponCount),
    totalPeople,
    armedPeople,
    weaponCount,
    riskScore,
    averageConfidence,
  };
};

const formatTime = (date: Date) =>
  date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

const getWebSocketUrl = () => {
  if (typeof window === 'undefined') {
    return 'ws://localhost:8000/stream';
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname || 'localhost';
  return `${protocol}//${host}:8000/stream`;
};

const mapFramePayload = (payload: FramePayload): FrameData => ({
  imageUrl: payload.image,
  annotatedImageUrl: payload.annotated_image,  // Model's visualization with boxes
  location: payload.source === 'demo' ? 'Demo thermal feed' : 'Drone thermal stream',
  detections: payload.detections.map((detection) => ({
    id: detection.id,
    x: detection.bbox.x,
    y: detection.bbox.y,
    width: detection.bbox.width,
    height: detection.bbox.height,
    confidence: Math.round(detection.confidence * 100),
    armed: detection.armed,
    weaponCount: detection.weapon_count,
    label: detection.label,
    kind: detection.kind,
  })),
  summary: {
    level: payload.summary.level,
    totalPeople: payload.summary.total_people,
    armedPeople: payload.summary.armed_people,
    weaponCount: payload.summary.weapon_count,
    riskScore: payload.summary.risk_score,
    averageConfidence: payload.summary.average_confidence,
  },
  timestamp: payload.timestamp,
  source: payload.source,
});

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === 'undefined') {
      return 'light';
    }

    return (window.localStorage.getItem('hawkeye-theme') as ThemeMode | null) ?? 'light';
  });
  const [currentFrame, setCurrentFrame] = useState<FrameData>(MOCK_FRAMES[0]);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [archive, setArchive] = useState<ArchiveEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<ThreatLevel | 'All'>('All');
  const [selectedArchiveEntry, setSelectedArchiveEntry] = useState<ArchiveEntry | null>(null);

  const summary = useMemo(() => currentFrame.summary ?? analyzeFrame(currentFrame), [currentFrame]);
  const levelStyle = LEVEL_STYLES[summary.level];

  const brief =
    summary.armedPeople === 0 && summary.totalPeople === 0
      ? 'No threats detected. Area is secure. Continue routine surveillance.'
      : summary.armedPeople === 0 && summary.totalPeople > 0
      ? `${summary.totalPeople} unarmed individual${summary.totalPeople > 1 ? 's' : ''} detected. Monitor for suspicious activity.`
      : `${summary.armedPeople} armed individual${summary.armedPeople > 1 ? 's' : ''} with ${summary.weaponCount} weapon${summary.weaponCount > 1 ? 's' : ''} detected. Immediate assessment required.`;
  const reviewStatus = summary.armedPeople > 0 ? 'Action Required' : 'Monitored';

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    window.localStorage.setItem('hawkeye-theme', theme);
  }, [theme]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let closedByUser = false;

    const connect = () => {
      const url = getWebSocketUrl();
      socket = new WebSocket(url);
      setConnectionStatus('Connecting...');
      setIsRefreshing(true);

      socket.onopen = () => {
        setConnectionStatus('Live');
        setIsRefreshing(false);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as { type: string; payload?: FramePayload; detail?: string };

          if (message.type === 'frame' && message.payload) {
            const nextFrame = mapFramePayload(message.payload);
            setCurrentFrame(nextFrame);
            setLastUpdate(new Date());
            setConnectionStatus('Live');
            setIsRefreshing(false);

            if (nextFrame.summary?.armedPeople > 0) {
              toast.error(`${nextFrame.summary.level} threat: human detected`, {
                description: `${nextFrame.summary.armedPeople} human${nextFrame.summary.armedPeople > 1 ? 's' : ''} and ${nextFrame.summary.weaponCount} visible weapon${nextFrame.summary.weaponCount > 1 ? 's' : ''} need review.`,
                icon: <AlertTriangle className="h-5 w-5" />,
              });
            }

            setArchive((previousArchive) => {
              const newEntry: ArchiveEntry = {
                id: `${nextFrame.source}-${nextFrame.timestamp ?? Date.now()}`,
                timestamp: new Date(nextFrame.timestamp ?? new Date().toISOString()),
                imageUrl: nextFrame.imageUrl,
                annotatedImageUrl: nextFrame.annotatedImageUrl,  // Store annotated image
                status: nextFrame.summary?.armedPeople && nextFrame.summary.armedPeople > 0 ? 'Action Required' : 'Monitored',
                ...nextFrame.summary,
              };

              return [newEntry, ...previousArchive].slice(0, 50);
            });
          }

          if (message.type === 'error') {
            setConnectionStatus('Error');
            toast.error(`Backend error: ${message.detail ?? 'Unable to process stream.'}`);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket frame', error);
        }
      };

      socket.onclose = () => {
        if (closedByUser) {
          return;
        }

        setConnectionStatus('Disconnected — reconnecting...');
        setIsRefreshing(false);
        reconnectTimer = window.setTimeout(connect, 3000);
      };

      socket.onerror = () => {
        setConnectionStatus('Connection error');
      };
    };

    connect();

    return () => {
      closedByUser = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }

      if (socket) {
        socket.close();
      }
    };
  }, []);

  const threatCounts = archive.reduce(
    (counts, entry) => ({
      ...counts,
      [entry.level]: counts[entry.level] + 1,
    }),
    { Clear: 0, Elevated: 0, High: 0, Critical: 0 } as Record<ThreatLevel, number>,
  );

  const filteredArchive = archive.filter((entry) => (filterLevel === 'All' ? true : entry.level === filterLevel));

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 selection:bg-blue-100 selection:text-blue-900 dark:bg-slate-950 dark:text-slate-100 dark:selection:bg-blue-900/60 dark:selection:text-blue-50">
      <Toaster position="top-right" richColors theme={theme} />

      <header className="border-b border-slate-200 bg-white/90 px-4 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-red-600 to-orange-600 text-white shadow-lg">
              <Eye className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight text-slate-950 dark:text-white">{APP_TITLE}</h1>
              <p className="text-sm text-slate-500 dark:text-slate-400">Autonomous thermal surveillance and real-time threat detection system.</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-sm">
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 font-medium text-slate-700 transition hover:border-blue-300 hover:text-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-blue-500 dark:hover:text-blue-300 dark:focus:ring-blue-950"
              onClick={() => setTheme((currentTheme) => (currentTheme === 'light' ? 'dark' : 'light'))}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
            >
              {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
              {theme === 'light' ? 'Dark mode' : 'Light mode'}
            </button>
            <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
              {connectionStatus}
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
              <Clock className="h-4 w-4" />
              Updated {formatTime(lastUpdate)}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 lg:px-6">
        <section className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900 dark:shadow-none">
          <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr] lg:items-center">
            <div>
              <span className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-sm font-medium text-blue-700 dark:bg-blue-950/50 dark:text-blue-300">
                <Eye className="h-4 w-4" />
                Human-centered threat review
              </span>
              <h2 className="mt-4 max-w-3xl text-3xl font-semibold tracking-tight text-slate-950 md:text-4xl dark:text-white">
                AI-powered thermal surveillance for enhanced security operations.
              </h2>
              <p className="mt-3 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-300">
                HawkEye AI utilizes advanced thermal imaging and deep learning to detect human thermal signatures and weapon profiles in real-time. The system provides immediate threat classification to enable rapid security response.
              </p>
            </div>

            <div className={`rounded-3xl border p-5 ${levelStyle.soft} ${levelStyle.border}`}>
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Current threat level</p>
                  <p className={`mt-1 text-3xl font-semibold ${levelStyle.text}`}>{summary.level}</p>
                </div>
                <span className={`rounded-full border px-3 py-1.5 text-sm font-semibold ${levelStyle.badge}`}>
                  {reviewStatus}
                </span>
              </div>
              <p className="mt-4 text-sm leading-6 text-slate-700 dark:text-slate-300">{brief}</p>
            </div>
          </div>
        </section>

        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard icon={<Users className="h-5 w-5" />} label="People detected" value={summary.totalPeople} />
          <MetricCard icon={<ShieldAlert className="h-5 w-5" />} label="Humans" value={summary.armedPeople} />
          <MetricCard icon={<AlertTriangle className="h-5 w-5" />} label="Visible weapons" value={summary.weaponCount} />
          <MetricCard icon={<CheckCircle className="h-5 w-5" />} label="Avg. confidence" value={`${summary.averageConfidence}%`} />
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <div className={cardClass}>
            <div className="flex flex-col gap-3 border-b border-slate-200 p-5 md:flex-row md:items-center md:justify-between dark:border-slate-800">
              <div>
              <h2 className="text-lg font-semibold text-slate-950 dark:text-white">Live camera frame</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">{currentFrame.location} - live thermal stream from backend</p>
            </div>
              <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                {isRefreshing && <RefreshCw className="h-4 w-4 animate-spin" />}
                <Camera className="h-4 w-4" />
                Channel 01
              </div>
            </div>

            <div className="relative h-[520px] overflow-hidden rounded-b-3xl bg-slate-100 dark:bg-slate-800">
              <img
                src={currentFrame.imageUrl}
                alt="Live aerial camera frame"
                className={`h-full w-full object-cover transition duration-500 ${isRefreshing ? 'opacity-75' : 'opacity-100'}`}
              />
              <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/55 to-transparent p-5">
                <div className="max-w-xl rounded-2xl bg-white/95 p-4 shadow-sm dark:bg-slate-900/95">
                  <p className="text-sm font-semibold text-slate-950 dark:text-white">Operator note</p>
                  <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">{brief}</p>
                </div>
              </div>
            </div>
          </div>

          <div className={cardClass}>
            <div className="border-b border-slate-200 p-5 dark:border-slate-800">
              <h2 className="text-lg font-semibold text-slate-950 dark:text-white">Detection details</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">People are labelled by armed status, weapons, and confidence.</p>
            </div>

            <div className="relative h-[520px] overflow-hidden rounded-b-3xl bg-slate-100 dark:bg-slate-800">
              <img src={currentFrame.annotatedImageUrl || currentFrame.imageUrl} alt="Detection review frame" className="h-full w-full object-cover" />

              <div className="absolute bottom-5 left-5 right-5 grid gap-3 sm:grid-cols-3">
                <InfoPill label="People detected" value={`${summary.totalPeople}`} />
                <InfoPill label="Weapons detected" value={`${summary.weaponCount}`} />
                <InfoPill label="Threat level" value={summary.level} />
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <div className={`${cardClass} p-5`}>
            <div className="mb-5 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-slate-950 dark:text-white">How threat is calculated</h2>
                <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Simple rules that are easy for reviewers to explain.</p>
              </div>
              <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${levelStyle.badge}`}>
                {summary.level}
              </span>
            </div>

            <div className="space-y-3 text-sm text-slate-600 dark:text-slate-300">
              <RuleRow label="No human detected" value="Clear" />
              <RuleRow label="Human detected, no weapon" value="Elevated" />
              <RuleRow label="Armed individual detected" value="High" />
              <RuleRow label="Multiple armed individuals" value="Critical" />
            </div>
          </div>

          <div className={`${cardClass} p-5`}>
            <div className="mb-5">
              <h2 className="text-lg font-semibold text-slate-950 dark:text-white">Archive summary</h2>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">A plain-language snapshot of recent frames.</p>
            </div>

            <div className="grid gap-3 sm:grid-cols-4">
              {(['Clear', 'Elevated', 'High', 'Critical'] as ThreatLevel[]).map((level) => (
                <div key={level} className={`rounded-2xl border p-4 ${LEVEL_STYLES[level].soft} ${LEVEL_STYLES[level].border}`}>
                  <p className={`text-2xl font-semibold ${LEVEL_STYLES[level].text}`}>{threatCounts[level]}</p>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">{level}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className={cardClass}>
          <div className="flex flex-col gap-4 border-b border-slate-200 p-5 md:flex-row md:items-center md:justify-between dark:border-slate-800">
            <div>
              <h2 className="text-lg font-semibold text-slate-950 dark:text-white">Detection archive</h2>
              <p className="text-sm text-slate-500 dark:text-slate-400">{archive.length} recent events saved for review.</p>
            </div>

            <div className="relative w-full md:w-auto">
              <Filter className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <select
                className="w-full appearance-none rounded-2xl border border-slate-200 bg-white py-2.5 pl-10 pr-8 text-sm text-slate-700 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100 md:w-[190px] dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:focus:ring-blue-950"
                value={filterLevel}
                onChange={(event) => setFilterLevel(event.target.value as ThreatLevel | 'All')}
              >
                <option value="All">All levels</option>
                <option value="Clear">Clear</option>
                <option value="Elevated">Elevated</option>
                <option value="High">High</option>
                <option value="Critical">Critical</option>
              </select>
            </div>
          </div>

          <div className="min-h-[300px] overflow-auto">
            {filteredArchive.length === 0 ? (
              <div className="flex min-h-[300px] flex-col items-center justify-center px-6 text-center text-slate-500 dark:text-slate-400">
                <div className="mb-4 grid h-14 w-14 place-items-center rounded-3xl bg-slate-100 dark:bg-slate-800">
                  <Search className="h-6 w-6" />
                </div>
                <p className="text-base font-medium text-slate-900 dark:text-white">No archived events yet</p>
                <p className="mt-1 max-w-md text-sm">
                  New frames will appear here with armed-person counts, visible weapons, and the current review level.
                </p>
              </div>
            ) : (
              <table className="w-full min-w-[860px] border-collapse text-left">
                <thead className="sticky top-0 z-10 border-b border-slate-200 bg-slate-50 text-xs uppercase tracking-wide text-slate-500 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-400">
                  <tr>
                    <th className="px-5 py-4 font-semibold">Time</th>
                    <th className="px-5 py-4 font-semibold">Snapshot</th>
                    <th className="px-5 py-4 font-semibold">Humans</th>
                    <th className="px-5 py-4 font-semibold">Weapons</th>
                    <th className="px-5 py-4 font-semibold">Level</th>
                    <th className="px-5 py-4 text-right font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 text-sm dark:divide-slate-800">
                  {filteredArchive.map((entry) => (
                    <tr 
                      key={entry.id} 
                      className="cursor-pointer transition hover:bg-slate-50 dark:hover:bg-slate-800/60"
                      onClick={() => setSelectedArchiveEntry(entry)}
                    >
                      <td className="px-5 py-4 text-slate-600 dark:text-slate-300">{formatTime(entry.timestamp)}</td>
                      <td className="px-5 py-4">
                        <div className="h-12 w-20 overflow-hidden rounded-2xl border border-slate-200 bg-slate-100 dark:border-slate-700 dark:bg-slate-800">
                          <img src={entry.imageUrl} alt="Archived detection snapshot" className="h-full w-full object-cover" />
                        </div>
                      </td>
                      <td className="px-5 py-4 text-slate-700 dark:text-slate-300">{entry.totalPeople}</td>
                      <td className="px-5 py-4 text-slate-700 dark:text-slate-300">{entry.weaponCount}</td>
                      <td className="px-5 py-4">
                        <span className={`rounded-full border px-3 py-1.5 text-xs font-semibold ${LEVEL_STYLES[entry.level].badge}`}>
                          {entry.level}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-right">
                        <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 dark:border-slate-700 dark:text-slate-300">
                          <CheckCircle className="h-3.5 w-3.5" />
                          {entry.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      </main>

      {/* Archive Detail Modal */}
      {selectedArchiveEntry && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 p-4 backdrop-blur-sm"
          onClick={() => setSelectedArchiveEntry(null)}
        >
          <div 
            className="relative w-full max-w-5xl max-h-[90vh] overflow-y-auto rounded-3xl border border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-slate-900"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button - Top Right Corner */}
            <button
              onClick={() => setSelectedArchiveEntry(null)}
              className="absolute right-4 top-4 z-10 flex h-10 w-10 items-center justify-center rounded-full border-2 border-slate-300 bg-white text-slate-700 shadow-lg transition hover:bg-slate-100 hover:border-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
              aria-label="Close modal"
            >
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Header */}
            <div className="border-b border-slate-200 p-6 pr-16 dark:border-slate-800">
              <h3 className="text-2xl font-semibold text-slate-950 dark:text-white">Detection Details</h3>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                Captured at {selectedArchiveEntry.timestamp.toLocaleString()}
              </p>
            </div>

            {/* Content */}
            <div className="p-6">
              {/* Threat Level Banner */}
              <div className={`mb-6 rounded-2xl border p-5 ${LEVEL_STYLES[selectedArchiveEntry.level].soft} ${LEVEL_STYLES[selectedArchiveEntry.level].border}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Threat Level</p>
                    <p className={`mt-1 text-3xl font-bold ${LEVEL_STYLES[selectedArchiveEntry.level].text}`}>
                      {selectedArchiveEntry.level}
                    </p>
                  </div>
                  <span className={`rounded-full border px-5 py-2.5 text-base font-semibold ${LEVEL_STYLES[selectedArchiveEntry.level].badge}`}>
                    Risk Score: {selectedArchiveEntry.riskScore}
                  </span>
                </div>
              </div>

              {/* Detection Stats - Above Image */}
              <div className="mb-6 grid gap-4 sm:grid-cols-4">
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800">
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">Total People</p>
                  <p className="mt-2 text-3xl font-bold text-slate-950 dark:text-white">
                    {selectedArchiveEntry.totalPeople}
                  </p>
                </div>
                <div className="rounded-2xl border border-orange-200 bg-orange-50 p-4 dark:border-orange-800 dark:bg-orange-950/30">
                  <p className="text-xs font-medium uppercase tracking-wide text-orange-600 dark:text-orange-400">Armed People</p>
                  <p className="mt-2 text-3xl font-bold text-orange-700 dark:text-orange-300">
                    {selectedArchiveEntry.armedPeople}
                  </p>
                </div>
                <div className="rounded-2xl border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950/30">
                  <p className="text-xs font-medium uppercase tracking-wide text-red-600 dark:text-red-400">Weapons</p>
                  <p className="mt-2 text-3xl font-bold text-red-700 dark:text-red-300">
                    {selectedArchiveEntry.weaponCount}
                  </p>
                </div>
                <div className="rounded-2xl border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-950/30">
                  <p className="text-xs font-medium uppercase tracking-wide text-blue-600 dark:text-blue-400">Confidence</p>
                  <p className="mt-2 text-3xl font-bold text-blue-700 dark:text-blue-300">
                    {selectedArchiveEntry.averageConfidence}%
                  </p>
                </div>
              </div>

              {/* Image with Bounding Boxes */}
              <div className="mb-6 overflow-hidden rounded-2xl border-2 border-slate-300 bg-slate-100 shadow-lg dark:border-slate-700 dark:bg-slate-800">
                <img 
                  src={selectedArchiveEntry.annotatedImageUrl || selectedArchiveEntry.imageUrl} 
                  alt="Detection frame with bounding boxes" 
                  className="w-full h-auto object-contain"
                  style={{ maxHeight: '500px' }}
                />
              </div>

              {/* Status */}
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5 dark:border-slate-700 dark:bg-slate-800">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Review Status</p>
                    <p className="mt-1 text-lg font-semibold text-slate-950 dark:text-white">
                      {selectedArchiveEntry.status}
                    </p>
                  </div>
                  <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-2.5 text-sm font-semibold text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
                    <CheckCircle className="h-5 w-5" />
                    {selectedArchiveEntry.status}
                  </span>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="border-t border-slate-200 p-6 dark:border-slate-800">
              <button
                onClick={() => setSelectedArchiveEntry(null)}
                className="w-full rounded-2xl bg-blue-600 px-6 py-3.5 text-base font-semibold text-white transition hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-200 dark:focus:ring-blue-900"
              >
                Close and Return to Dashboard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className={`${cardClass} p-5`}>
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-2xl bg-blue-50 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300">{icon}</div>
      <p className="text-3xl font-semibold text-slate-950 dark:text-white">{value}</p>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{label}</p>
    </div>
  );
}

function InfoPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-white/95 p-3 shadow-sm dark:bg-slate-900/95">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400 dark:text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-semibold text-slate-900 dark:text-white">{value}</p>
    </div>
  );
}

function RuleRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-2xl bg-slate-50 px-4 py-3 dark:bg-slate-800">
      <span>{label}</span>
      <span className="font-semibold text-slate-900 dark:text-white">{value}</span>
    </div>
  );
}
