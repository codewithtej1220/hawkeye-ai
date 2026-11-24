import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, ClipboardList, Eye, Moon, Radio, ScanSearch, Sun } from 'lucide-react';
import { Toaster, toast } from 'sonner';
import AnalyzeMedia from './AnalyzeMedia';
import EventLog from './EventLog';
import LiveOps from './LiveOps';
import {
  LEVEL_RANK,
  LEVEL_STYLES,
  LivePayload,
  LogEvent,
  StreamInfo,
  ThreatLevel,
  describeSituation,
  formatClock,
  getApiBase,
  getWebSocketUrl,
} from './shared';

type ThemeMode = 'light' | 'dark';
type TabId = 'live' | 'analyze' | 'log';
type Connection = 'connecting' | 'online' | 'offline';

const ALERT_REPEAT_MS = 15000;
const STALE_AFTER_MS = 5000;

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'live', label: 'Live operations', icon: <Radio className="h-4 w-4" /> },
  { id: 'analyze', label: 'Analyze media', icon: <ScanSearch className="h-4 w-4" /> },
  { id: 'log', label: 'Event log', icon: <ClipboardList className="h-4 w-4" /> },
];

interface Health {
  detector_ready: boolean;
  model?: string | null;
  detail?: string | null;
  log_events?: number | null;
}

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(() => (window.localStorage.getItem('hawkeye-theme') as ThemeMode | null) ?? 'dark');
  const [tab, setTab] = useState<TabId>('live');
  const [frame, setFrame] = useState<LivePayload | null>(null);
  const [stream, setStream] = useState<StreamInfo | null>(null);
  const [connection, setConnection] = useState<Connection>('connecting');
  const [health, setHealth] = useState<Health | null>(null);
  const [latestEvent, setLatestEvent] = useState<LogEvent | null>(null);
  const [unseenEvents, setUnseenEvents] = useState(0);
  // Alerts are throttled: the stream sends several frames per second
  const lastAlertRef = useRef<{ level: ThreatLevel; at: number }>({ level: 'Clear', at: 0 });
  const lastFrameAtRef = useRef(0);
  const tabRef = useRef<TabId>('live');

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    window.localStorage.setItem('hawkeye-theme', theme);
  }, [theme]);

  useEffect(() => {
    tabRef.current = tab;
    if (tab === 'log') {
      setUnseenEvents(0);
    }
  }, [tab]);

  // Live connection: frames, new log entries
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let closed = false;

    const connect = () => {
      setConnection('connecting');
      socket = new WebSocket(getWebSocketUrl());
      socket.onopen = () => setConnection('online');
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as { type: string; payload?: unknown };
          if (message.type === 'frame' && message.payload) {
            const payload = message.payload as LivePayload;
            setFrame(payload);
            lastFrameAtRef.current = Date.now();
            if (payload.stream) {
              setStream(payload.stream);
            }
            alertOnThreat(payload);
          } else if (message.type === 'log' && message.payload) {
            const logEvent = message.payload as LogEvent;
            setLatestEvent(logEvent);
            if (tabRef.current !== 'log') {
              setUnseenEvents((n) => n + 1);
            }
          } else if (message.type === 'log_update' && message.payload) {
            setLatestEvent({ ...(message.payload as LogEvent) });
          }
        } catch (error) {
          console.error('Failed to parse a message from the backend', error);
        }
      };
      socket.onclose = () => {
        if (closed) {
          return;
        }
        setConnection('offline');
        reconnectTimer = window.setTimeout(connect, 3000);
      };
      socket.onerror = () => setConnection('offline');
    };

    const alertOnThreat = (payload: LivePayload) => {
      const summary = payload.summary;
      const last = lastAlertRef.current;
      const now = Date.now();
      const isThreat = summary.armed_people > 0 || summary.weapon_count > 0;
      // Alert when the threat escalates, or remind at most every ALERT_REPEAT_MS while it persists
      if (isThreat && (LEVEL_RANK[summary.level] > LEVEL_RANK[last.level] || now - last.at > ALERT_REPEAT_MS)) {
        toast.error(`${summary.level} threat: ${summary.armed_people > 0 ? 'armed person' : 'weapon'} detected`, {
          description: describeSituation(summary.total_people, summary.armed_people, summary.weapon_count, summary.possible_armed ?? 0),
          icon: <AlertTriangle className="h-5 w-5" />,
        });
        lastAlertRef.current = { level: summary.level, at: now };
      } else if (!isThreat && now - last.at > ALERT_REPEAT_MS) {
        lastAlertRef.current = { level: 'Clear', at: last.at };
      }
    };

    connect();
    return () => {
      closed = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      socket?.close();
    };
  }, []);

  // Stream health: mark the stream as stopped when frames stop arriving; poll the backend status
  useEffect(() => {
    const staleTimer = window.setInterval(() => {
      if (lastFrameAtRef.current && Date.now() - lastFrameAtRef.current > STALE_AFTER_MS) {
        setStream((current) => (current && current.live ? { ...current, live: false, fps: 0 } : current));
      }
    }, 1000);
    const loadHealth = async () => {
      try {
        const response = await fetch(`${getApiBase()}/health`);
        if (response.ok) {
          setHealth((await response.json()) as Health);
        }
      } catch {
        setHealth(null);
      }
    };
    void loadHealth();
    const healthTimer = window.setInterval(loadHealth, 10000);
    return () => {
      window.clearInterval(staleTimer);
      window.clearInterval(healthTimer);
    };
  }, []);

  const level = frame?.summary.level ?? 'Clear';
  const streaming = Boolean(stream?.live);

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 selection:bg-blue-100 selection:text-blue-900 dark:bg-slate-950 dark:text-slate-100 dark:selection:bg-blue-900/60 dark:selection:text-blue-50">
      <Toaster position="top-right" richColors theme={theme} />

      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-3 px-4 py-3 lg:flex-row lg:items-center lg:justify-between lg:px-6">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-red-600 to-orange-500 text-white shadow">
              <Eye className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight tracking-tight text-slate-950 dark:text-white">HawkEye AI</h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">Drone surveillance · people and weapon detection</p>
            </div>
          </div>

          <nav className="flex items-center gap-1 rounded-2xl border border-slate-200 bg-slate-50 p-1 dark:border-slate-800 dark:bg-slate-900" aria-label="Sections">
            {TABS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setTab(item.id)}
                aria-current={tab === item.id ? 'page' : undefined}
                className={`relative inline-flex items-center gap-2 rounded-xl px-3.5 py-2 text-sm font-medium transition focus:outline-none focus-visible:ring-4 focus-visible:ring-blue-200 dark:focus-visible:ring-blue-900 ${
                  tab === item.id
                    ? 'bg-white text-slate-950 shadow-sm dark:bg-slate-800 dark:text-white'
                    : 'text-slate-600 hover:text-slate-950 dark:text-slate-400 dark:hover:text-white'
                }`}
              >
                {item.icon}
                {item.label}
                {item.id === 'log' && unseenEvents > 0 && (
                  <span className="rounded-full bg-orange-600 px-1.5 text-[11px] font-semibold leading-5 text-white">{unseenEvents > 99 ? '99+' : unseenEvents}</span>
                )}
                {item.id === 'live' && tab !== 'live' && streaming && (
                  <span className={`h-2 w-2 rounded-full ${LEVEL_STYLES[level].bar}`} title={`Threat level: ${level}`} />
                )}
              </button>
            ))}
          </nav>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <StatusPill
              ok={connection === 'online'}
              warn={connection === 'connecting'}
              label={connection === 'online' ? 'Backend online' : connection === 'connecting' ? 'Connecting...' : 'Backend offline'}
              title={health?.detail ?? undefined}
            />
            <StatusPill
              ok={streaming}
              label={streaming ? `Drone stream ${stream?.fps ?? 0} fps` : frame ? `Stream stopped ${formatClock(frame.timestamp)}` : 'No drone stream'}
            />
            <StatusPill
              ok={Boolean(health?.detector_ready)}
              warn={!health}
              label={health?.detector_ready ? 'Detector ready' : health ? 'Detector not ready' : 'Detector unknown'}
              title={health?.model ?? health?.detail ?? undefined}
            />
            <button
              type="button"
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700 transition hover:border-blue-300 focus:outline-none focus-visible:ring-4 focus-visible:ring-blue-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-blue-500 dark:focus-visible:ring-blue-900"
              onClick={() => setTheme((current) => (current === 'light' ? 'dark' : 'light'))}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
            >
              {theme === 'light' ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
              {theme === 'light' ? 'Dark' : 'Light'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1600px] px-4 py-5 lg:px-6">
        {connection === 'offline' && (
          <div className="mb-5 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
            The dashboard cannot reach the backend at {getApiBase()}. Start it (RUN_ALL_HAWKEYE.bat); the page reconnects by itself.
          </div>
        )}
        {/* All three stay mounted so an analysis in progress or log filters survive switching tabs */}
        <div className={tab === 'live' ? '' : 'hidden'}>
          <LiveOps frame={frame} stream={stream} connected={connection === 'online'} />
        </div>
        <div className={tab === 'analyze' ? '' : 'hidden'}>
          <AnalyzeMedia />
        </div>
        <div className={tab === 'log' ? '' : 'hidden'}>
          <EventLog latestEvent={latestEvent} />
        </div>
      </main>

      <footer className="mx-auto w-full max-w-[1600px] px-4 pb-6 text-xs text-slate-500 lg:px-6 dark:text-slate-500">
        {health?.model ? `Models: ${health.model}` : 'Models: not loaded'}
        {typeof health?.log_events === 'number' ? ` · ${health.log_events} events on record` : ''}
      </footer>
    </div>
  );
}

function StatusPill({ ok, warn, label, title }: { ok: boolean; warn?: boolean; label: string; title?: string }) {
  const dot = ok ? 'bg-emerald-500' : warn ? 'bg-amber-500' : 'bg-red-500';
  return (
    <span title={title} className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">
      <span className={`h-2 w-2 rounded-full ${dot} ${ok ? 'animate-pulse' : ''}`} />
      {label}
    </span>
  );
}
