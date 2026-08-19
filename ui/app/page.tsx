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
  num_candidates: number;
  num_results: number;
  weight_clip: number;
  weight_asr: number;
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
  thumbnail_url?: string;
};

type SearchResponse = {
  query: string;
  hits: SearchHit[];
};

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api"
).replace(/\/$/, "");

const DEFAULT_PARAMETERS: SearchParameters = {
  num_candidates: 500,
  num_results: 100,
  weight_clip: 0.8,
  weight_asr: 0.2,
  delta: 3,
};

const PRESETS: Array<{
  label: string;
  description: string;
  visual: number;
  asr: number;
}> = [
  { label: "Cân bằng", description: "80% visual · 20% ASR", visual: 0.8, asr: 0.2 },
  { label: "Hình ảnh", description: "Chỉ CLIP + SigLIP2", visual: 1, asr: 0 },
  { label: "Lời thoại", description: "60% visual · 40% ASR", visual: 0.6, asr: 0.4 },
];

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "--:--";
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder
    .toFixed(1)
    .padStart(4, "0")}`;
}

function assertSearchResponse(value: unknown): SearchResponse {
  if (!value || typeof value !== "object") {
    throw new Error("Backend trả về dữ liệu không hợp lệ.");
  }
  const response = value as Partial<SearchResponse>;
  if (typeof response.query !== "string" || !Array.isArray(response.hits)) {
    throw new Error("Backend trả về response không đúng schema /search.");
  }
  return response as SearchResponse;
}

export default function Home() {
  const [query, setQuery] = useState("");
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
  const controllerRef = useRef<AbortController | null>(null);

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

  function updateParameter<K extends keyof SearchParameters>(
    key: K,
    value: SearchParameters[K],
  ) {
    setParameters((current) => ({ ...current, [key]: value }));
  }

  async function runSearch() {
    const normalizedQuery = query.trim();
    if (!normalizedQuery || isSearching) return;

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setIsSearching(true);
    setError(null);
    setSelectedHit(null);
    const startedAt = performance.now();

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: normalizedQuery, ...parameters }),
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
      setLastQuery(data.query);
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

  async function copyAnswer(hit: SearchHit) {
    const value = `${hit.video_id}, ${hit.frame_idx}`;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(value);
      window.setTimeout(() => setCopied(null), 1600);
    } catch {
      setError("Trình duyệt không cho phép sao chép vào clipboard.");
    }
  }

  function exportCsv() {
    if (!hits.length) return;
    const header = "rank,video_id,frame_idx,pts_time,score";
    const rows = hits.map((hit) =>
      [hit.rank, hit.video_id, hit.frame_idx, hit.pts_time, hit.score].join(","),
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
          <span className="mode-chip">Visual + ASR</span>
        </div>

        <form className="query-form" onSubmit={handleSubmit}>
          <label htmlFor="search-query" className="sr-only">
            Mô tả sự kiện cần tìm
          </label>
          <textarea
            id="search-query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleQueryKeyDown}
            placeholder="Mô tả cảnh, hành động, nhân vật hoặc lời thoại cần tìm…"
            rows={3}
          />
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
              {parameters.num_candidates} candidates · {parameters.num_results} results
            </span>
          </summary>
          <div className="settings-content">
            <div className="preset-group" aria-label="Preset trọng số">
              {PRESETS.map((preset) => {
                const active =
                  parameters.weight_clip === preset.visual &&
                  parameters.weight_asr === preset.asr;
                return (
                  <button
                    key={preset.label}
                    className={active ? "preset active" : "preset"}
                    type="button"
                    onClick={() =>
                      setParameters((current) => ({
                        ...current,
                        weight_clip: preset.visual,
                        weight_asr: preset.asr,
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
                <span>Ứng viên mỗi index</span>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={parameters.num_candidates}
                  onChange={(event) =>
                    updateParameter("num_candidates", Number(event.target.value))
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
                  value={parameters.weight_clip}
                  onChange={(event) =>
                    updateParameter("weight_clip", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>Trọng số ASR</span>
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={parameters.weight_asr}
                  onChange={(event) =>
                    updateParameter("weight_asr", Number(event.target.value))
                  }
                />
              </label>
              <label>
                <span>ASR window (giây)</span>
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
                  ? "Thử mô tả lại sự kiện hoặc điều chỉnh trọng số visual và ASR."
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
                        {hit.thumbnail_url ? (
                          // The current API has no image URL; this is ready for a future media endpoint.
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={hit.thumbnail_url} alt={`Keyframe ${hit.frame_idx}`} />
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
                        <span>Frame {hit.frame_idx.toLocaleString("vi-VN")}</span>
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
                <div className="inspector-frame frame-placeholder">
                  <span>{selectedHit.video_id}</span>
                  <strong>{formatTime(selectedHit.pts_time)}</strong>
                </div>
                <dl>
                  <div>
                    <dt>Video ID</dt>
                    <dd>{selectedHit.video_id}</dd>
                  </div>
                  <div>
                    <dt>Frame ID</dt>
                    <dd>{selectedHit.frame_idx.toLocaleString("vi-VN")}</dd>
                  </div>
                  <div>
                    <dt>Timestamp</dt>
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
                </dl>
                <button
                  className="primary-inspector-action"
                  type="button"
                  onClick={() => void copyAnswer(selectedHit)}
                >
                  Chép đáp án KIS
                  <span>{selectedHit.video_id}, {selectedHit.frame_idx}</span>
                </button>
                <p className="inspector-note">
                  Backend hiện chưa cung cấp ảnh keyframe hoặc video preview. Inspector
                  đã sẵn sàng nhận media khi endpoint được bổ sung.
                </p>
              </aside>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
