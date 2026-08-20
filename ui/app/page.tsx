"use client";

import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

type HealthState = "checking" | "online" | "offline";

type SearchParameters = {
  num_candidates_visual: number;
  num_results: number;
  weight_visual: number;
  weight_transcript: number;
  weight_sem_text: number;
  weight_bm25: number;
  bm25_top_segments: number;
  sem_top_segments: number;
  delta: number;
};

type SearchHit = {
  rank: number;
  score: number;
  clip_row: number;
  video_id: string;
  pts_time: number;
  row_idx_in_video: number;
  frame_idx: number;
  fps: number;
  source: string;
  thumbnail_url?: string | null;
  video_url?: string | null;
};

type SearchResponse = {
  query_vi: string;
  query_en: string;
  query_en_source: "user" | "translated";
  hits: SearchHit[];
};

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api"
).replace(/\/$/, "");

const DEFAULT_PARAMETERS: SearchParameters = {
  num_candidates_visual: 500,
  num_results: 100,
  weight_visual: 0.8,
  weight_transcript: 0.2,
  weight_sem_text: 0.6,
  weight_bm25: 0.4,
  bm25_top_segments: 50,
  sem_top_segments: 50,
  delta: 1,
};

const PRESETS: Array<{
  label: string;
  description: string;
  visual: number;
  transcript: number;
  semantic: number;
  bm25: number;
}> = [
  {
    label: "Cân bằng",
    description: "80% visual · 20% transcript",
    visual: 0.8,
    transcript: 0.2,
    semantic: 0.6,
    bm25: 0.4,
  },
  {
    label: "Hình ảnh",
    description: "Chỉ CLIP + SigLIP2",
    visual: 1,
    transcript: 0,
    semantic: 0.6,
    bm25: 0.4,
  },
  {
    label: "Lời thoại",
    description: "60% visual · 40% transcript",
    visual: 0.6,
    transcript: 0.4,
    semantic: 0.6,
    bm25: 0.4,
  },
];

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "--:--";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder
    .toFixed(1)
    .padStart(4, "0")}`;
}

function formatOffset(seconds: number) {
  if (!Number.isFinite(seconds) || Math.abs(seconds) < 0.05) {
    return "Đúng keyframe gốc";
  }
  return `${seconds > 0 ? "+" : ""}${seconds.toFixed(1)} giây so với keyframe`;
}

function resolveApiUrl(path: string) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

function assertSearchResponse(value: unknown): SearchResponse {
  if (!value || typeof value !== "object") {
    throw new Error("Backend trả về dữ liệu không hợp lệ.");
  }
  const response = value as Partial<SearchResponse>;
  if (
    typeof response.query_vi !== "string" ||
    typeof response.query_en !== "string" ||
    (response.query_en_source !== "user" &&
      response.query_en_source !== "translated") ||
    !Array.isArray(response.hits)
  ) {
    throw new Error("Backend trả về response không đúng schema /search.");
  }
  return response as SearchResponse;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [queryEn, setQueryEn] = useState("");
  const [queryEnSource, setQueryEnSource] = useState<
    "user" | "translated" | null
  >(null);
  const [lastQuery, setLastQuery] = useState("");
  const [parameters, setParameters] =
    useState<SearchParameters>(DEFAULT_PARAMETERS);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [health, setHealth] = useState<HealthState>("checking");
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [playbackPosition, setPlaybackPosition] = useState<{
    clipRow: number;
    seconds: number;
  } | null>(null);
  const [failedThumbnails, setFailedThumbnails] = useState<Set<string>>(
    () => new Set(),
  );
  const controllerRef = useRef<AbortController | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/check_health`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error("Health check failed");
      setHealth("online");
    } catch {
      setHealth("offline");
    }
  }, []);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE_URL}/check_health`, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error("Health check failed");
        if (active) setHealth("online");
      })
      .catch(() => {
        if (active) setHealth("offline");
      });
    return () => {
      active = false;
      controllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !selectedHit?.video_url) return;

    const seekToKeyframe = () => {
      const upperBound = Number.isFinite(video.duration)
        ? video.duration
        : selectedHit.pts_time;
      video.currentTime = Math.max(
        0,
        Math.min(selectedHit.pts_time, upperBound),
      );
      video.pause();
    };

    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      seekToKeyframe();
      return;
    }

    video.addEventListener("loadedmetadata", seekToKeyframe, { once: true });
    return () => video.removeEventListener("loadedmetadata", seekToKeyframe);
  }, [selectedHit]);

  function updateParameter<K extends keyof SearchParameters>(
    key: K,
    value: SearchParameters[K],
  ) {
    setParameters((current) => ({ ...current, [key]: value }));
  }

  async function runSearch() {
    const normalizedQuery = query.trim();
    if (!normalizedQuery || isSearching) return;
    if (parameters.weight_visual + parameters.weight_transcript <= 0) {
      setError("Cần đặt trọng số dương cho visual hoặc transcript.");
      return;
    }
    if (
      parameters.weight_transcript > 0 &&
      parameters.weight_sem_text + parameters.weight_bm25 <= 0
    ) {
      setError("Transcript cần trọng số dương cho MiniLM hoặc BM25.");
      return;
    }

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setIsSearching(true);
    setError(null);
    setSelectedHit(null);
    setFailedThumbnails(new Set());
    const startedAt = performance.now();

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_vi: normalizedQuery,
          query_en:
            queryEnSource === "translated" ? null : (queryEn.trim() || null),
          ...parameters,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(
          detail || `Search thất bại với mã HTTP ${response.status}.`,
        );
      }

      const data = assertSearchResponse(await response.json());
      setHits(data.hits);
      setLastQuery(data.query_vi);
      setQueryEn(data.query_en);
      setQueryEnSource(data.query_en_source);
      setSelectedHit(data.hits[0] ?? null);
      setElapsedMs(Math.round(performance.now() - startedAt));
      setHealth("online");
    } catch (searchError) {
      if (searchError instanceof DOMException && searchError.name === "AbortError") {
        return;
      }
      setError(
        searchError instanceof Error
          ? searchError.message
          : "Không thể kết nối tới search backend.",
      );
      setHealth("offline");
      setElapsedMs(null);
    } finally {
      if (controllerRef.current === controller) {
        setIsSearching(false);
      }
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runSearch();
  }

  function handleQueryKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      void runSearch();
    }
  }

  function thumbnailUrl(hit: SearchHit) {
    if (!hit.thumbnail_url || failedThumbnails.has(hit.thumbnail_url)) return null;
    return resolveApiUrl(hit.thumbnail_url);
  }

  function markThumbnailFailed(hit: SearchHit) {
    const url = hit.thumbnail_url;
    if (!url) return;
    setFailedThumbnails((current) => {
      const next = new Set(current);
      next.add(url);
      return next;
    });
  }

  function moveVideoBy(seconds: number) {
    const video = videoRef.current;
    if (!video || !selectedHit) return;
    const upperBound = Number.isFinite(video.duration)
      ? video.duration
      : Number.POSITIVE_INFINITY;
    video.currentTime = Math.max(
      0,
      Math.min(video.currentTime + seconds, upperBound),
    );
    setPlaybackPosition({
      clipRow: selectedHit.clip_row,
      seconds: video.currentTime,
    });
  }

  function jumpToKeyframe() {
    const video = videoRef.current;
    if (!video || !selectedHit) return;
    const upperBound = Number.isFinite(video.duration)
      ? video.duration
      : selectedHit.pts_time;
    video.currentTime = Math.max(
      0,
      Math.min(selectedHit.pts_time, upperBound),
    );
    video.pause();
    setPlaybackPosition({
      clipRow: selectedHit.clip_row,
      seconds: video.currentTime,
    });
  }

  function trackPlaybackPosition(video: HTMLVideoElement) {
    if (!selectedHit) return;
    setPlaybackPosition({
      clipRow: selectedHit.clip_row,
      seconds: video.currentTime,
    });
  }

  async function copyKisAnswer(videoId: string, frameIdx: number) {
    const value = `${videoId}, ${frameIdx}`;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(value);
      window.setTimeout(() => setCopied(null), 1600);
    } catch {
      setError("Trình duyệt không cho phép sao chép vào clipboard.");
    }
  }

  function copyAnswer(hit: SearchHit) {
    return copyKisAnswer(hit.video_id, hit.frame_idx);
  }

  function exportCsv() {
    if (!hits.length) return;
    const header = "rank,video_id,frame_idx,pts_time,fps,source,score";
    const rows = hits.map((hit) =>
      [
        hit.rank,
        hit.video_id,
        hit.frame_idx,
        hit.pts_time,
        hit.fps,
        hit.source,
        hit.score,
      ].join(","),
    );
    const blob = new Blob([[header, ...rows].join("\n")], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `pacs-search-${Date.now()}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const statusText = {
    checking: "Đang kiểm tra backend",
    online: "Backend sẵn sàng",
    offline: "Backend chưa kết nối",
  }[health];
  const selectedThumbnailUrl = selectedHit ? thumbnailUrl(selectedHit) : null;
  const selectedVideoUrl = selectedHit?.video_url
    ? resolveApiUrl(selectedHit.video_url)
    : null;
  const playbackSeconds = selectedHit && playbackPosition?.clipRow === selectedHit.clip_row
    ? playbackPosition.seconds
    : (selectedHit?.pts_time ?? 0);
  const selectedFps = selectedHit && Number.isFinite(selectedHit.fps) && selectedHit.fps > 0
    ? selectedHit.fps
    : null;
  const playbackFrameIdx = selectedHit
    ? Math.max(
        0,
        selectedFps
          ? selectedHit.frame_idx + Math.round(
              (playbackSeconds - selectedHit.pts_time) * selectedFps,
            )
          : selectedHit.frame_idx,
      )
    : 0;
  const playbackOffset = selectedHit
    ? playbackSeconds - selectedHit.pts_time
    : 0;

  return (
    <main className="workspace-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">P</span>
          <div>
            <p className="eyebrow">AIC 2026 · Retrieval console</p>
            <h1>PACs Search</h1>
          </div>
        </div>
        <button
          className={`system-status status-${health}`}
          type="button"
          onClick={() => {
            setHealth("checking");
            void checkHealth();
          }}
          title={`${API_BASE_URL} · Nhấn để kiểm tra lại`}
        >
          <span className="status-dot" />
          <span>{statusText}</span>
        </button>
      </header>

      <section className="query-section" aria-labelledby="search-heading">
        <div className="query-heading">
          <div>
            <p className="eyebrow">Textual KIS</p>
            <h2 id="search-heading">Tìm khoảnh khắc trong video</h2>
          </div>
          <span className="mode-chip">Visual + MiniLM + BM25</span>
        </div>

        <form className="query-form" onSubmit={handleSubmit}>
          <div className="query-fields">
            <label className="query-field" htmlFor="search-query-vi">
              <span className="query-field-label">
                Tiếng Việt
                <small>SigLIP2 · MiniLM · BM25</small>
              </span>
              <textarea
                id="search-query-vi"
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  if (queryEnSource === "translated") {
                    setQueryEn("");
                    setQueryEnSource(null);
                  }
                }}
                onKeyDown={handleQueryKeyDown}
                placeholder="Mô tả cảnh, hành động, nhân vật hoặc lời thoại cần tìm…"
                rows={3}
              />
            </label>
            <label className="query-field" htmlFor="search-query-en">
              <span className="query-field-label">
                Tiếng Anh cho CLIP
                <small>
                  {queryEnSource === "translated"
                    ? "Bản dịch tự động · có thể chỉnh sửa"
                    : queryEnSource === "user"
                      ? "Bản tiếng Anh do bạn nhập"
                      : "Để trống để hệ thống tự dịch"}
                </small>
              </span>
              <textarea
                id="search-query-en"
                value={queryEn}
                onChange={(event) => {
                  setQueryEn(event.target.value);
                  setQueryEnSource(event.target.value.trim() ? "user" : null);
                }}
                onKeyDown={handleQueryKeyDown}
                placeholder="Optional English query for CLIP…"
                rows={3}
              />
            </label>
          </div>
          <div className="query-actions">
            <p>
              <kbd>Ctrl</kbd> + <kbd>Enter</kbd> để tìm kiếm
            </p>
            <button type="submit" disabled={!query.trim() || isSearching}>
              {isSearching ? "Đang truy xuất…" : "Tìm kiếm"}
              <span aria-hidden="true">{isSearching ? "···" : "→"}</span>
            </button>
          </div>
        </form>

        <details className="search-settings">
          <summary>
            <span>Tùy chỉnh retrieval</span>
            <span className="settings-summary">
              {parameters.num_candidates_visual} visual · {parameters.num_results} results
            </span>
          </summary>
          <div className="settings-content">
            <div className="preset-group" aria-label="Preset trọng số">
              {PRESETS.map((preset) => {
                const active =
                  parameters.weight_visual === preset.visual &&
                  parameters.weight_transcript === preset.transcript &&
                  parameters.weight_sem_text === preset.semantic &&
                  parameters.weight_bm25 === preset.bm25;
                return (
                  <button
                    key={preset.label}
                    className={active ? "preset active" : "preset"}
                    type="button"
                    onClick={() =>
                      setParameters((current) => ({
                        ...current,
                        weight_visual: preset.visual,
                        weight_transcript: preset.transcript,
                        weight_sem_text: preset.semantic,
                        weight_bm25: preset.bm25,
                      }))
                    }
                  >
                    <strong>{preset.label}</strong>
                    <span>{preset.description}</span>
                  </button>
                );
              })}
            </div>
            <div className="parameter-grid">
              <label>
                <span>Ứng viên visual/index</span>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={parameters.num_candidates_visual}
                  onChange={(event) =>
                    updateParameter("num_candidates_visual", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>Số kết quả</span>
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={parameters.num_results}
                  onChange={(event) =>
                    updateParameter("num_results", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>Trọng số visual</span>
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={parameters.weight_visual}
                  onChange={(event) =>
                    updateParameter("weight_visual", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>Trọng số transcript</span>
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={parameters.weight_transcript}
                  onChange={(event) =>
                    updateParameter("weight_transcript", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>MiniLM trong transcript</span>
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={parameters.weight_sem_text}
                  onChange={(event) =>
                    updateParameter("weight_sem_text", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>BM25 trong transcript</span>
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={parameters.weight_bm25}
                  onChange={(event) =>
                    updateParameter("weight_bm25", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>Top đoạn BM25</span>
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={parameters.bm25_top_segments}
                  onChange={(event) =>
                    updateParameter("bm25_top_segments", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>Top đoạn MiniLM</span>
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={parameters.sem_top_segments}
                  onChange={(event) =>
                    updateParameter("sem_top_segments", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>Transcript window (giây)</span>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  value={parameters.delta}
                  onChange={(event) =>
                    updateParameter("delta", Number(event.target.value))
                  }
                />
              </label>
            </div>
          </div>
        </details>
      </section>

      <section className="results-section" aria-labelledby="results-heading">
        <div className="section-header">
          <div>
            <p className="eyebrow">Ranked candidates</p>
            <h2 id="results-heading">Kết quả truy xuất</h2>
          </div>
          <div className="result-tools">
            {elapsedMs !== null && <span>{elapsedMs.toLocaleString("vi-VN")} ms</span>}
            <span className="result-count">{hits.length} kết quả</span>
            <button type="button" onClick={exportCsv} disabled={!hits.length}>
              Xuất CSV
            </button>
          </div>
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <div>
              <strong>Không thể hoàn thành truy vấn</strong>
              <p>{error}</p>
            </div>
            <button type="button" onClick={() => void runSearch()}>
              Thử lại
            </button>
          </div>
        )}

        {isSearching && (
          <div className="loading-grid" aria-label="Đang tải kết quả">
            {Array.from({ length: 8 }, (_, index) => (
              <div className="loading-card" key={index} />
            ))}
          </div>
        )}

        {!isSearching && !error && hits.length === 0 && (
          <div className="empty-state">
            <div className="empty-visual" aria-hidden="true">
              <span>01</span>
              <span>02</span>
              <span>03</span>
            </div>
            <div>
              <h3>{lastQuery ? "Không tìm thấy candidate" : "Sẵn sàng cho truy vấn đầu tiên"}</h3>
              <p>
                {lastQuery
                  ? "Thử mô tả lại sự kiện hoặc điều chỉnh fusion visual và transcript."
                  : "Nhập mô tả sự kiện. Hệ thống sẽ xếp hạng các keyframe phù hợp bằng CLIP, SigLIP2 và tín hiệu lời thoại."}
              </p>
            </div>
          </div>
        )}

        {!isSearching && hits.length > 0 && (
          <div className={selectedHit ? "results-layout has-inspector" : "results-layout"}>
            <div className="results-grid">
              {hits.map((hit) => {
                const selected = selectedHit?.clip_row === hit.clip_row;
                const resolvedThumbnailUrl = thumbnailUrl(hit);
                return (
                  <article
                    className={selected ? "result-card selected" : "result-card"}
                    key={`${hit.clip_row}-${hit.rank}`}
                  >
                    <button
                      className="card-select"
                      type="button"
                      aria-label={`Xem kết quả hạng ${hit.rank}`}
                      onClick={() => setSelectedHit(hit)}
                    >
                      <div className="frame-preview">
                        {resolvedThumbnailUrl ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={resolvedThumbnailUrl}
                            alt={`Keyframe ${hit.frame_idx} của ${hit.video_id}`}
                            loading="lazy"
                            onError={() => markThumbnailFailed(hit)}
                          />
                        ) : (
                          <div className="frame-placeholder">
                            <span>{hit.video_id}</span>
                            <strong>{formatTime(hit.pts_time)}</strong>
                          </div>
                        )}
                        <span className="rank-badge">#{hit.rank}</span>
                        <span className="score-badge">{hit.score.toFixed(3)}</span>
                      </div>
                    </button>
                    <div className="card-meta">
                      <div>
                        <strong>{hit.video_id}</strong>
                        <span>
                          Frame {hit.frame_idx.toLocaleString("vi-VN")} · {hit.source}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          void copyAnswer(hit);
                        }}
                      >
                        {copied === `${hit.video_id}, ${hit.frame_idx}` ? "Đã chép" : "Chép"}
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>

            {selectedHit && (
              <aside className="inspector" aria-label="Chi tiết candidate">
                <div className="inspector-header">
                  <div>
                    <p className="eyebrow">Candidate inspector</p>
                    <h3>Hạng #{selectedHit.rank}</h3>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedHit(null)}
                    aria-label="Đóng inspector"
                  >
                    ×
                  </button>
                </div>
                <div className="inspector-frame">
                  {selectedVideoUrl ? (
                    // Source dataset does not include timed caption tracks.
                    // eslint-disable-next-line jsx-a11y/media-has-caption
                    <video
                      key={selectedHit.video_id}
                      ref={videoRef}
                      src={selectedVideoUrl}
                      poster={selectedThumbnailUrl ?? undefined}
                      preload="metadata"
                      controls
                      playsInline
                      onTimeUpdate={(event) => trackPlaybackPosition(event.currentTarget)}
                      onSeeked={(event) => trackPlaybackPosition(event.currentTarget)}
                      aria-label={`Video ${selectedHit.video_id}, bắt đầu tại ${formatTime(selectedHit.pts_time)}`}
                    />
                  ) : selectedThumbnailUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={selectedThumbnailUrl}
                      alt={`Keyframe ${selectedHit.frame_idx} của ${selectedHit.video_id}`}
                      onError={() => markThumbnailFailed(selectedHit)}
                    />
                  ) : (
                    <div className="frame-placeholder">
                      <span>{selectedHit.video_id}</span>
                      <strong>{formatTime(selectedHit.pts_time)}</strong>
                    </div>
                  )}
                </div>
                {selectedVideoUrl && (
                  <>
                    <div className="video-context-controls" aria-label="Điều khiển quanh keyframe">
                      <button type="button" onClick={() => moveVideoBy(-5)}>
                        −5 giây
                      </button>
                      <button type="button" onClick={jumpToKeyframe}>
                        Về keyframe · {formatTime(selectedHit.pts_time)}
                      </button>
                      <button type="button" onClick={() => moveVideoBy(5)}>
                        +5 giây
                      </button>
                    </div>
                    <section className="playhead-panel" aria-label="Vị trí video đang xem">
                      <div className="playhead-heading">
                        <div>
                          <p className="eyebrow">Playhead hiện tại</p>
                          <strong>{formatTime(playbackSeconds)}</strong>
                        </div>
                        <span>{formatOffset(playbackOffset)}</span>
                      </div>
                      <div className="playhead-values">
                        <div>
                          <span>Timestamp chính xác</span>
                          <strong>{playbackSeconds.toFixed(3)} giây</strong>
                        </div>
                        <div>
                          <span>Frame ID hiện tại</span>
                          <strong>{playbackFrameIdx.toLocaleString("vi-VN")}</strong>
                        </div>
                        <div>
                          <span>FPS nguồn</span>
                          <strong>{selectedFps?.toFixed(3) ?? "Không có"}</strong>
                        </div>
                      </div>
                      <button
                        className="copy-playhead-action"
                        type="button"
                        onClick={() => void copyKisAnswer(selectedHit.video_id, playbackFrameIdx)}
                      >
                        <span>
                          {copied === `${selectedHit.video_id}, ${playbackFrameIdx}`
                            ? "Đã chép đáp án đang xem"
                            : "Chép đáp án tại playhead"}
                        </span>
                        <code>{selectedHit.video_id}, {playbackFrameIdx}</code>
                      </button>
                    </section>
                  </>
                )}
                <dl>
                  <div>
                    <dt>Video ID</dt>
                    <dd>{selectedHit.video_id}</dd>
                  </div>
                  <div>
                    <dt>Frame ID retrieval</dt>
                    <dd>{selectedHit.frame_idx.toLocaleString("vi-VN")}</dd>
                  </div>
                  <div>
                    <dt>Timestamp retrieval</dt>
                    <dd>{selectedHit.pts_time.toFixed(3)} giây</dd>
                  </div>
                  <div>
                    <dt>Keyframe row</dt>
                    <dd>{selectedHit.row_idx_in_video}</dd>
                  </div>
                  <div>
                    <dt>FAISS row</dt>
                    <dd>{selectedHit.clip_row}</dd>
                  </div>
                  <div>
                    <dt>Retrieval score</dt>
                    <dd>{selectedHit.score.toFixed(6)}</dd>
                  </div>
                  <div>
                    <dt>Nguồn candidate</dt>
                    <dd>{selectedHit.source}</dd>
                  </div>
                </dl>
                <button
                  className="primary-inspector-action"
                  type="button"
                  onClick={() => void copyAnswer(selectedHit)}
                >
                  Chép đáp án KIS gốc
                  <span>{selectedHit.video_id}, {selectedHit.frame_idx}</span>
                </button>
                <p className="inspector-note">
                  Video đầy đủ được tải theo từng đoạn cần xem. Player mở và tạm dừng
                  tại keyframe, nhưng không giới hạn phạm vi tua hoặc phát.
                </p>
              </aside>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
