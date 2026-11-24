import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Cpu, FileImage, FileVideo, Flame, Link2, Loader2, ScanSearch, Upload, X } from 'lucide-react';
import { LEVEL_STYLES, METHOD_OPTIONS, MethodId, SensingInfo, ThreatLevel, cardClass, describeThreat, getApiBase } from './shared';

interface Summary {
  level: ThreatLevel;
  total_people: number;
  armed_people: number;
  weapon_count: number;
  risk_score: number;
  average_confidence: number;
  possible_armed?: number;
}

interface Detection {
  id: string;
  label: string;
  kind: 'person' | 'weapon' | 'other';
  confidence: number;
  armed: boolean;
  weapon_count: number;
  evidence?: string | null; // weapons: detector | zoom | classifier
  concealed?: boolean;
  concealment?: string | null;
  armed_evidence?: string | null; // people: detector | zoom | classifier | tracked
  armed_frames?: string | null;
  armed_confirmed?: boolean;
  armed_possible?: boolean;
}

/** Plain-language description of one detection for the result caption. */
const describeDetection = (d: Detection): string | null => {
  const pct = `${Math.round(d.confidence * 100)}%`;
  if (d.kind === 'weapon') {
    if (d.evidence === 'classifier') return null; // described on the person
    return `${d.concealed ? 'CONCEALED weapon' : d.evidence === 'zoom' ? 'weapon (zoomed look)' : 'weapon'} ${pct}`;
  }
  if (d.kind === 'person') {
    if (!d.armed) return `person ${pct}`;
    if (d.armed_possible) return `POSSIBLE weapon on distant person (verify${d.armed_frames ? `, ${d.armed_frames} frames` : ''}) ${pct}`;
    if (d.armed_evidence === 'classifier') return `armed person (distant, confirmed ${d.armed_frames} frames) ${pct}`;
    if (d.armed_evidence === 'tracked') return `armed person (confirmed over time, ${d.armed_frames} frames) ${pct}`;
    return `armed person${d.armed_confirmed ? ` (${d.armed_frames} frames)` : ''} ${pct}`;
  }
  return `${d.label} ${pct}`;
};

interface MediaFrame extends SensingInfo {
  time_s: number | null;
  frame_index: number;
  summary: Summary;
  detections: Detection[];
  annotated_image: string;
}

interface TimelineEntry {
  time_s: number | null;
  frame_index: number;
  level: ThreatLevel;
  total_people: number;
  armed_people: number;
  weapon_count: number;
  confirmed: boolean;
}

interface MediaAnalysis {
  kind: 'image' | 'video';
  filename: string;
  width: number;
  height: number;
  summary: Summary;
  frames_analyzed: number;
  duration_s: number | null;
  processing_ms: number;
  key_frames: MediaFrame[];
  timeline: TimelineEntry[];
  unconfirmed_weapon_frames: number;
  sensors: string[];
  methods_used: Record<string, number>;
}

const ACCEPT = 'image/*,video/*,.mp4,.avi,.mov,.mkv,.webm,.tif,.tiff';
const MAX_MB = 500;

const formatBytes = (bytes: number) =>
  bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

const formatSeconds = (seconds: number | null) => {
  if (seconds === null) return '';
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  // Tenths matter: video is sampled every 0.5 s
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
};

const errorText = async (response: Response) => {
  try {
    const body = await response.json();
    return typeof body.detail === 'string' ? body.detail : `Request failed (HTTP ${response.status}).`;
  } catch {
    return `Request failed (HTTP ${response.status}).`;
  }
};

export default function AnalyzeMedia() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [url, setUrl] = useState('');
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<false | 'upload' | 'url'>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MediaAnalysis | null>(null);
  const [selected, setSelected] = useState(0);
  const [method, setMethod] = useState<MethodId>('auto');
  const [distant, setDistant] = useState<'off' | 'flag' | 'alert'>('flag');
  const [thermalFile, setThermalFile] = useState<File | null>(null);
  const thermalInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const chooseFile = (candidate: File | undefined) => {
    if (!candidate) return;
    const isMedia = candidate.type.startsWith('image/') || candidate.type.startsWith('video/') ||
      /\.(mp4|avi|mov|mkv|webm|m4v|wmv|mpe?g|tiff?)$/i.test(candidate.name);
    if (!isMedia) {
      setError('Please choose an image or video file.');
      return;
    }
    if (candidate.size > MAX_MB * 1024 * 1024) {
      setError(`File is larger than the ${MAX_MB} MB limit.`);
      return;
    }
    setError(null);
    setResult(null);
    setFile(candidate);
  };

  const run = async (mode: 'upload' | 'url') => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(mode);
    setError(null);
    setResult(null);

    try {
      let response: Response;
      if (mode === 'upload' && file) {
        const form = new FormData();
        form.append('file', file);
        form.append('method', method);
        form.append('distant', distant);
        if (thermalFile) {
          form.append('thermal', thermalFile);
        }
        response = await fetch(`${getApiBase()}/analyze/upload`, { method: 'POST', body: form, signal: controller.signal });
      } else {
        response = await fetch(`${getApiBase()}/analyze/url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url.trim(), method: method === 'rgbt' ? 'auto' : method }),
          signal: controller.signal,
        });
      }
      if (!response.ok) {
        throw new Error(await errorText(response));
      }
      const analysis = (await response.json()) as MediaAnalysis;
      setResult(analysis);
      // Open on the most threatening key frame
      const worst = analysis.key_frames.reduce((best, frame, index) =>
        frame.summary.risk_score > analysis.key_frames[best].summary.risk_score ? index : best, 0);
      setSelected(worst);
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      const message = (err as Error).message;
      setError(message === 'Failed to fetch'
        ? `Cannot reach the HawkEye backend at ${getApiBase()}. Make sure it is running.`
        : message);
    } finally {
      if (abortRef.current === controller) {
        setBusy(false);
      }
    }
  };

  const cancel = () => {
    abortRef.current?.abort();
    setBusy(false);
  };

  const reset = () => {
    cancel();
    setFile(null);
    setThermalFile(null);
    setUrl('');
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
    if (thermalInputRef.current) thermalInputRef.current.value = '';
  };

  const isVideo = file ? file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm|m4v|wmv|mpe?g)$/i.test(file.name) : false;
  const needsPair = method === 'rgbt' && !thermalFile;
  const frame = result?.key_frames[selected];
  const summary = result?.summary;
  const style = summary ? LEVEL_STYLES[summary.level] : null;

  return (
    <section className={cardClass} aria-labelledby="analyze-media-heading">
      <div className="flex flex-col gap-2 border-b border-slate-200 p-5 md:flex-row md:items-center md:justify-between dark:border-slate-800">
        <div>
          <h2 id="analyze-media-heading" className="flex items-center gap-2 text-lg font-semibold text-slate-950 dark:text-white">
            <ScanSearch className="h-5 w-5" />
            Analyze media
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Upload a photo or video from this device, or paste a direct link. Works with thermal, colour and greyscale footage.
          </p>
        </div>
        {(file || result || url) && (
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-2 self-start rounded-full border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-600 transition hover:bg-slate-50 md:self-auto dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            <X className="h-4 w-4" />
            Clear
          </button>
        )}
      </div>

      <div className="grid gap-6 p-5 lg:grid-cols-[0.9fr_1.1fr]">
        {/* ---------------- input ---------------- */}
        <div className="flex flex-col gap-4">
          <div
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => (event.key === 'Enter' || event.key === ' ') && inputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              chooseFile(event.dataTransfer.files?.[0]);
            }}
            className={`flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-3 overflow-hidden rounded-2xl border-2 border-dashed p-4 text-center transition focus:outline-none focus:ring-4 focus:ring-blue-100 dark:focus:ring-blue-950 ${
              dragging
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30'
                : 'border-slate-300 bg-slate-50 hover:border-blue-400 dark:border-slate-700 dark:bg-slate-950/40 dark:hover:border-blue-500'
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              className="hidden"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
            {file && previewUrl ? (
              <>
                {isVideo ? (
                  <video src={previewUrl} className="max-h-[200px] rounded-xl" muted controls onClick={(e) => e.stopPropagation()} />
                ) : (
                  <img src={previewUrl} alt="Selected upload preview" className="max-h-[200px] rounded-xl object-contain" />
                )}
                <p className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
                  {isVideo ? <FileVideo className="h-4 w-4" /> : <FileImage className="h-4 w-4" />}
                  <span className="max-w-[260px] truncate font-medium">{file.name}</span>
                  <span className="text-slate-400">· {formatBytes(file.size)}</span>
                </p>
              </>
            ) : (
              <>
                <div className="grid h-12 w-12 place-items-center rounded-2xl bg-white text-blue-600 shadow-sm dark:bg-slate-900 dark:text-blue-300">
                  <Upload className="h-6 w-6" />
                </div>
                <p className="text-sm font-medium text-slate-800 dark:text-slate-100">Drop an image or video here, or click to browse</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">JPG, PNG, WEBP, TIFF · MP4, AVI, MOV, MKV, WEBM · up to {MAX_MB} MB</p>
              </>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-slate-200 px-3 py-2 text-sm dark:border-slate-800">
            <Flame className="h-4 w-4 text-orange-500" />
            <span className="text-slate-600 dark:text-slate-300">Thermal file of the same scene</span>
            <span className="text-xs text-slate-400">(optional · enables RGBT fusion)</span>
            <input
              ref={thermalInputRef}
              type="file"
              accept={ACCEPT}
              className="hidden"
              onChange={(event) => {
                setThermalFile(event.target.files?.[0] ?? null);
                setResult(null);
              }}
            />
            {thermalFile ? (
              <span className="ml-auto flex items-center gap-2">
                <span className="max-w-[160px] truncate font-medium text-slate-700 dark:text-slate-200">{thermalFile.name}</span>
                <button
                  type="button"
                  aria-label="Remove thermal file"
                  onClick={() => {
                    setThermalFile(null);
                    if (thermalInputRef.current) thermalInputRef.current.value = '';
                  }}
                  className="rounded-full p-1 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </span>
            ) : (
              <button
                type="button"
                onClick={() => thermalInputRef.current?.click()}
                className="ml-auto rounded-full border border-slate-200 px-3 py-1 text-xs font-semibold text-slate-600 hover:border-orange-300 hover:text-orange-700 dark:border-slate-700 dark:text-slate-300"
              >
                Add thermal
              </button>
            )}
          </div>

          <fieldset>
            <legend className="mb-1.5 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
              <Cpu className="h-3.5 w-3.5" />
              Sensing method
            </legend>
            <div className="flex flex-wrap gap-1.5" role="radiogroup">
              {METHOD_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  role="radio"
                  aria-checked={method === option.id}
                  title={option.help}
                  onClick={() => setMethod(option.id)}
                  className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition ${
                    method === option.id
                      ? 'border-blue-500 bg-blue-600 text-white'
                      : 'border-slate-200 text-slate-600 hover:border-blue-300 hover:text-blue-700 dark:border-slate-700 dark:text-slate-300'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <p className="mt-1.5 text-xs text-slate-500 dark:text-slate-400">
              {needsPair
                ? 'RGBT fusion needs both files: the visible-light one above and a thermal file of the same scene.'
                : METHOD_OPTIONS.find((option) => option.id === method)?.help}
            </p>
          </fieldset>

          <fieldset>
            <legend className="mb-1.5 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
              Distant-weapon search
            </legend>
            <div className="flex flex-wrap gap-1.5" role="radiogroup">
              {([
                ['off', 'Off', 'Full frame + zoom only. No extra false alarms, but distant weapons are mostly missed.'],
                ['flag', 'Flag', 'Also searches for small distant people and judges if they are armed. Findings are shown as "possible weapon" to verify; they do not raise the threat level.'],
                ['alert', 'Alert', 'As Flag, but a distant finding confirmed over several video frames raises the threat to High. More false alarms.'],
              ] as const).map(([id, label, help]) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={distant === id}
                  title={help}
                  onClick={() => setDistant(id)}
                  className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition ${
                    distant === id
                      ? 'border-amber-500 bg-amber-500 text-white'
                      : 'border-slate-200 text-slate-600 hover:border-amber-300 hover:text-amber-700 dark:border-slate-700 dark:text-slate-300'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </fieldset>

          <button
            type="button"
            disabled={!file || !!busy || needsPair}
            onClick={() => run('upload')}
            className="inline-flex items-center justify-center gap-2 rounded-2xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-200 disabled:cursor-not-allowed disabled:bg-slate-300 dark:focus:ring-blue-900 dark:disabled:bg-slate-700"
          >
            {busy === 'upload' ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanSearch className="h-4 w-4" />}
            {busy === 'upload' ? (isVideo ? 'Analyzing video…' : 'Analyzing image…') : 'Analyze file'}
          </button>

          <div className="flex items-center gap-3 text-xs uppercase tracking-wide text-slate-400">
            <span className="h-px flex-1 bg-slate-200 dark:bg-slate-800" />
            or from a link
            <span className="h-px flex-1 bg-slate-200 dark:bg-slate-800" />
          </div>

          <form
            className="flex flex-col gap-2 sm:flex-row"
            onSubmit={(event) => {
              event.preventDefault();
              if (url.trim() && !busy) run('url');
            }}
          >
            <div className="relative flex-1">
              <Link2 className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://example.com/footage.mp4"
                aria-label="Direct link to an image or video"
                className="w-full rounded-2xl border border-slate-200 bg-white py-2.5 pl-10 pr-3 text-sm text-slate-800 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:focus:ring-blue-950"
              />
            </div>
            <button
              type="submit"
              disabled={!url.trim() || !!busy}
              className="inline-flex items-center justify-center gap-2 rounded-2xl border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
            >
              {busy === 'url' && <Loader2 className="h-4 w-4 animate-spin" />}
              Analyze link
            </button>
          </form>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Links must point straight to the media file (ending in .jpg, .mp4, …), not to a web page or video site.
          </p>

          {busy && (
            <div className="flex items-center justify-between gap-3 rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800 dark:border-blue-900 dark:bg-blue-950/30 dark:text-blue-200">
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {busy === 'url' ? 'Downloading and analyzing…' : isVideo ? 'Sampling the video and checking each moment for people and weapons…' : 'Detecting people and weapons…'}
              </span>
              <button type="button" onClick={cancel} className="font-semibold underline-offset-2 hover:underline">
                Cancel
              </button>
            </div>
          )}

          {error && (
            <div role="alert" className="flex items-start gap-2 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* ---------------- results ---------------- */}
        <div className="flex min-h-[300px] flex-col gap-4">
          {!result || !summary || !style ? (
            <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-400">
              <ScanSearch className="mb-3 h-8 w-8 text-slate-400" />
              Results appear here: threat level, people and weapons found, and annotated frames.
            </div>
          ) : (
            <>
              <div className={`rounded-2xl border p-4 ${style.soft} ${style.border}`}>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      {result.kind === 'video' ? 'Highest threat in video' : 'Threat level'}
                    </p>
                    <p className={`text-2xl font-semibold ${style.text}`}>{summary.level}</p>
                  </div>
                  <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${style.badge}`}>
                    {summary.armed_people > 0 || summary.weapon_count > 0 ? 'Action Required' : 'Monitored'}
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-700 dark:text-slate-300">
                  {describeThreat(summary.total_people, summary.armed_people, summary.weapon_count)}
                </p>
                {frame?.detections.some((d) => d.concealed) && (
                  <p className="mt-2 rounded-xl bg-red-600/10 px-3 py-2 text-sm font-semibold text-red-700 dark:text-red-300">
                    Concealed weapon: {frame.detections.find((d) => d.concealed)?.concealment}
                  </p>
                )}
                {frame?.detections.some((d) => d.armed_evidence === 'classifier') && (
                  <p className="mt-2 text-xs text-slate-600 dark:text-slate-300">
                    Distant person judged armed by the second-stage classifier: the weapon is too small to outline, so the whole person is marked.
                    {(summary.possible_armed ?? 0) > 0 && ' Seen in a single frame only, so it is reported as a possible weapon to verify; in video it is confirmed once seen in 2 of 6 frames.'}
                  </p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="People" value={summary.total_people} />
                <Stat label="Armed" value={summary.armed_people} />
                <Stat label="Weapons" value={summary.weapon_count} />
                <Stat label="Avg. confidence" value={`${summary.average_confidence}%`} />
              </div>

              {frame && <SensingPanel info={frame} methodsUsed={result.kind === 'video' ? result.methods_used : undefined} />}

              {frame && (
                <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-950 dark:border-slate-800">
                  <img src={frame.annotated_image} alt="Annotated analysis result" className="max-h-[420px] w-full object-contain" />
                  <div className="flex flex-wrap items-center justify-between gap-2 bg-white px-4 py-2 text-xs text-slate-500 dark:bg-slate-900 dark:text-slate-400">
                    <span>
                      {result.kind === 'video' ? `At ${formatSeconds(frame.time_s)} · ` : ''}
                      <span className={`font-semibold ${LEVEL_STYLES[frame.summary.level].text}`}>{frame.summary.level}</span>
                      {' · '}
                      {frame.detections.length === 0
                        ? 'nothing detected'
                        : frame.detections.map(describeDetection).filter(Boolean).join(', ')}
                    </span>
                    <span>
                      {result.width}×{result.height} · {result.frames_analyzed} frame{result.frames_analyzed === 1 ? '' : 's'} analysed · {(result.processing_ms / 1000).toFixed(1)} s
                    </span>
                  </div>
                </div>
              )}

              {result.kind === 'video' && (
                <>
                  <div>
                    <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      Threat timeline · {formatSeconds(result.duration_s)}
                    </p>
                    <div className="flex h-4 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800" role="img" aria-label="Threat level over the video">
                      {result.timeline.map((entry) => (
                        <div
                          key={entry.frame_index}
                          className={`h-full flex-1 ${LEVEL_STYLES[entry.level].bar} ${entry.confirmed ? '' : 'opacity-40'}`}
                          title={`${formatSeconds(entry.time_s)} · ${entry.level} · ${entry.total_people} people, ${entry.weapon_count} weapons${entry.confirmed ? '' : ' (unconfirmed: single frame)'}`}
                        />
                      ))}
                    </div>
                    {result.unconfirmed_weapon_frames > 0 && (
                      <p className="mt-1.5 text-xs text-slate-500 dark:text-slate-400">
                        {result.unconfirmed_weapon_frames} brief weapon detection{result.unconfirmed_weapon_frames === 1 ? '' : 's'} (seen in a single frame, shown faded) not counted toward the verdict. Review them manually if needed.
                      </p>
                    )}
                  </div>

                  <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                    {result.key_frames.map((keyFrame, index) => (
                      <button
                        key={keyFrame.frame_index}
                        type="button"
                        onClick={() => setSelected(index)}
                        className={`overflow-hidden rounded-xl border-2 text-left transition ${
                          index === selected ? 'border-blue-500' : `${LEVEL_STYLES[keyFrame.summary.level].border} hover:border-blue-300`
                        }`}
                      >
                        <img src={keyFrame.annotated_image} alt={`Key moment at ${formatSeconds(keyFrame.time_s)}`} className="aspect-video w-full object-cover" />
                        <span className={`block px-1.5 py-1 text-[11px] font-semibold ${LEVEL_STYLES[keyFrame.summary.level].text}`}>
                          {formatSeconds(keyFrame.time_s)} · {keyFrame.summary.level}
                        </span>
                      </button>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function SensingPanel({ info, methodsUsed }: { info: SensingInfo; methodsUsed?: Record<string, number> }) {
  const env = info.environment;
  const label = (id: string) => METHOD_OPTIONS.find((option) => option.id === id)?.label ?? id;
  return (
    <div className="rounded-2xl border border-slate-200 p-3 text-sm dark:border-slate-800">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">Sensing method</span>
        <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-semibold text-blue-700 dark:bg-blue-950/50 dark:text-blue-300">
          {info.method_label ?? label(info.method ?? '')}
        </span>
        {methodsUsed && Object.keys(methodsUsed).length > 1 && (
          <span className="text-xs text-slate-500 dark:text-slate-400">
            across the video: {Object.entries(methodsUsed).map(([id, n]) => `${label(id)} ${n}`).join(' · ')}
          </span>
        )}
      </div>
      {info.method_reasons && info.method_reasons.length > 0 && (
        <p className="mt-1.5 text-xs text-slate-600 dark:text-slate-300">{info.method_reasons.join('; ')}</p>
      )}
      {env && (
        <div className="mt-2 flex flex-wrap gap-1.5 text-[11px]">
          {env.rgb_reliability != null && <Chip label="RGB reliability" value={`${Math.round(env.rgb_reliability * 100)}%`} />}
          {env.thermal_reliability != null && <Chip label="Thermal reliability" value={`${Math.round(env.thermal_reliability * 100)}%`} />}
          {env.rgb_brightness != null && <Chip label="Brightness" value={`${Math.round(env.rgb_brightness)}`} />}
          {env.thermal_contrast != null && <Chip label="Thermal contrast" value={`${Math.round(env.thermal_contrast)}`} />}
          {env.conditions.map((condition) => <Chip key={condition} label="Conditions" value={condition} />)}
        </div>
      )}
    </div>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full border border-slate-200 px-2 py-0.5 text-slate-600 dark:border-slate-700 dark:text-slate-300">
      <span className="text-slate-400">{label}:</span> {value}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-800 dark:bg-slate-950/40">
      <p className="text-xl font-semibold text-slate-950 dark:text-white">{value}</p>
      <p className="text-xs text-slate-500 dark:text-slate-400">{label}</p>
    </div>
  );
}
