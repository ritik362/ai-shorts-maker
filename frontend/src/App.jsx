
import { useState, useRef, useEffect } from "react";
import "./App.css";

const BACKEND_URL = "http://127.0.0.1:8000";
const ALLOWED_TYPES = ["video/mp4", "video/quicktime", "video/webm"];
const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".webm"];

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [clipCount, setClipCount] = useState(10);
  const [status, setStatus] = useState("");

  // Upload states
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadError, setUploadError] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);

  const fileInputRef = useRef(null);
  const pollTimerRef = useRef(null);

  // Permission & Pipeline states
  const [hasPermission, setHasPermission] = useState(false);
  const [stage, setStage] = useState("idle"); // idle | validating | downloading | analyzing | processing | completed | error
  const [progress, setProgress] = useState(0);
  const [currentClip, setCurrentClip] = useState(0);
  const [totalClips, setTotalClips] = useState(0);
  const [generatedClips, setGeneratedClips] = useState([]);
  const [activeTaskId, setActiveTaskId] = useState(null);
  const [videoErrors, setVideoErrors] = useState({});

  // Stop polling helper
  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  // Clean up polling on unmount
  useEffect(() => {
    return () => stopPolling();
  }, []);

  // ── File Validation & Processing ──────────────────────────────
  const processFile = (file) => {
    setUploadError("");

    if (!file) return;

    // BUG-FIX 1: some OS/browsers report MOV as "" or non-standard types,
    // so always fall back to extension check as primary gate.
    const ext = "." + file.name.split(".").pop().toLowerCase();
    const isValidType =
      ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTENSIONS.includes(ext);

    if (!isValidType) {
      setUploadError(
        `❌ Unsupported file "${file.name}". Please upload an MP4, MOV, or WebM video.`
      );
      return;
    }

    // BUG-FIX 2: revoke BEFORE creating a new object URL so memory is freed
    // even when previewUrl state hasn't updated yet (stale closure guard).
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    setUploadedFile(file);
    // BUG-FIX 3: clear any previous status message when a new file is chosen
    setStatus("");
  };

  const handleFileChange = (e) => {
    processFile(e.target.files[0]);
    // Reset input so the same file can be re-selected after removal
    e.target.value = "";
  };

  const handleRemoveFile = () => {
    // BUG-FIX 5: use functional setter to safely revoke regardless of closure timing
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setUploadedFile(null);
    setUploadError("");
    setStatus("");
  };

  // ── Drag & Drop ───────────────────────────────────────────────
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    // BUG-FIX 4: only clear drag-over when leaving the zone itself,
    // not when moving over a child element inside the zone.
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragOver(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    processFile(file);
  };

  // ── Generate — 6-Stage Automated Pipeline ───────────────────
  const [isLoading, setIsLoading] = useState(false);

  const startPolling = (taskId) => {
    stopPolling();
    setActiveTaskId(taskId);

    pollTimerRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/process-status/${taskId}`);
        if (!res.ok) {
          throw new Error(`Failed to fetch status (HTTP ${res.status})`);
        }
        const data = await res.json();

        setProgress(data.progress || 0);
        setCurrentClip(data.current_clip || 0);
        setTotalClips(data.total_clips || 0);

        if (data.status === "completed") {
          stopPolling();
          setIsLoading(false);
          setStage("completed");
          setGeneratedClips(data.clips || []);
          setStatus(`🎉 Success! All ${data.clips.length} Shorts generated successfully.`);
        } else if (data.status === "failed") {
          stopPolling();
          setIsLoading(false);
          setStage("error");
          setGeneratedClips(data.clips || []); // preserve any partial clips
          setStatus(`❌ Processing failed: ${data.error || "Unknown error"}`);
        } else if (data.status === "analyzing_highlights") {
          setStage("analyzing");
          setStatus("AI analyzing video for top highlight moments & key dialogues...");
        } else if (data.status === "processing") {
          setStage("processing");
          setStatus(
            `Rendering Clip ${data.current_clip} of ${data.total_clips} (${data.progress}% complete)...`
          );
        } else if (data.status === "queued") {
          setStage("processing");
          setStatus("Job queued. Waiting for sequential worker to start rendering...");
        }
      } catch (err) {
        stopPolling();
        setIsLoading(false);
        setStage("error");
        setStatus(`❌ Error checking task status: ${err.message}`);
      }
    }, 2500);
  };

  const handleGenerate = async () => {
    // If a local file is selected
    if (uploadedFile) {
      setStatus(
        `File "${uploadedFile.name}" selected! Local file processing will be connected next. Please use a YouTube URL for full automated generation.`
      );
      return;
    }

    if (!videoUrl.trim()) {
      setStatus("Please paste a YouTube video link first.");
      return;
    }

    if (!hasPermission) {
      setStatus("⚠️ Please confirm that you own or have permission to process this video.");
      return;
    }

    // Reset pipeline state
    stopPolling();
    setIsLoading(true);
    setStage("validating");
    setStatus("Step 1/4: Validating video URL...");
    setProgress(0);
    setCurrentClip(0);
    setTotalClips(clipCount);
    setGeneratedClips([]);
    setVideoErrors({});

    try {
      // ── Stage 1: Validate URL ───────────────────────────────
      const valRes = await fetch(`${BACKEND_URL}/validate-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoUrl.trim() }),
      });
      const valData = await valRes.json();

      if (!valRes.ok || !valData.valid) {
        throw new Error(valData.reason || valData.detail || "Invalid video URL.");
      }
      if (valData.platform !== "youtube") {
        throw new Error("Only YouTube video processing is supported currently.");
      }

      // ── Stage 2: Download Video ─────────────────────────────
      setStage("downloading");
      setStatus("Step 2/4: Downloading video from YouTube... (this may take a few moments)");

      const dlRes = await fetch(`${BACKEND_URL}/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: valData.url,
          confirmed_permission: true,
        }),
      });
      const dlData = await dlRes.json();

      if (!dlRes.ok || dlData.status !== "success") {
        throw new Error(dlData.reason || dlData.detail || "Download failed.");
      }

      const downloadedFilename = dlData.filename;

      // ── Stage 3: Inspect Video Info ─────────────────────────
      setStage("analyzing");
      setStatus("Step 3/4: Analyzing video length and safe clip limits...");

      const infoRes = await fetch(
        `${BACKEND_URL}/video-info/${encodeURIComponent(downloadedFilename)}`
      );
      const infoData = await infoRes.json();

      if (!infoRes.ok || !infoData.is_supported) {
        throw new Error(infoData.message || "Video is not supported for Shorts clipping.");
      }

      // Clamp requested clip count to maximum non-overlapping clips
      let targetClipCount = clipCount;
      if (infoData.max_clip_count < clipCount) {
        targetClipCount = infoData.max_clip_count;
        setStatus(
          `Video duration is ${infoData.duration_formatted}. Automatically clamped to ${targetClipCount} non-overlapping clips.`
        );
      }
      setTotalClips(targetClipCount);

      // ── Stage 4: Queue Background Processing ───────────────
      setStage("processing");
      setStatus(`Step 4/4: Queuing ${targetClipCount} Shorts for generation...`);

      const procRes = await fetch(`${BACKEND_URL}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: downloadedFilename,
          clip_count: targetClipCount,
          clip_duration: 25,
          detection_mode: "ai",
        }),
      });
      const procData = await procRes.json();

      if (!procRes.ok) {
        throw new Error(procData.detail || "Failed to start background clipping job.");
      }

      // ── Stage 5: Start Polling Status ───────────────────────
      startPolling(procData.task_id);
    } catch (err) {
      stopPolling();
      setIsLoading(false);
      setStage("error");
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setStatus("❌ Cannot connect to backend. Please ensure FastAPI is running on port 8000.");
      } else {
        setStatus(`❌ ${err.message || "An unexpected error occurred."}`);
      }
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <div className="logo">Clip<span>AI</span></div>
        <div className="nav-label">AI VIDEO STUDIO</div>
      </header>

      <main className="container">
        <section className="hero">
          <div className="tag">✦ YOUR PERSONAL AI VIDEO EDITOR</div>
          <h1>
            Turn Long Videos Into
            <span> Viral Shorts.</span>
          </h1>
          <p>
            Paste your YouTube or Instagram video link, or upload a video file.
            Create short videos with automatic captions.
          </p>
        </section>

        <section className="generator">
          <h2>Start Creating Shorts</h2>
          <p className="subtext">
            Add your video link or upload a local file below.
          </p>

          {/* ── Video URL ── */}
          <label>VIDEO URL</label>
          <input
            type="url"
            placeholder="Paste YouTube or Instagram link..."
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
          />

          {/* ── Divider ── */}
          <div className="or-divider">
            <span>OR</span>
          </div>

          {/* ── Upload Area ── */}
          <label>UPLOAD VIDEO FILE</label>

          {!uploadedFile ? (
            <div
              className={`upload-zone ${isDragOver ? "drag-over" : ""}`}
              onClick={() => fileInputRef.current.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && fileInputRef.current.click()}
              aria-label="Upload video file"
            >
              <div className="upload-icon">🎬</div>
              <p className="upload-title">Drag &amp; drop your video here</p>
              <p className="upload-sub">or click to browse files</p>
              <div className="upload-formats">MP4 &nbsp;·&nbsp; MOV &nbsp;·&nbsp; WebM</div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
            </div>
          ) : (
            <div className="upload-preview">
              {/* Video Player */}
              <video
                className="video-preview"
                src={previewUrl}
                controls
                preload="metadata"
              />

              {/* File Info + Remove */}
              <div className="file-info">
                <div className="file-details">
                  <span className="file-icon">🎥</span>
                  <div>
                    <p className="file-name">{uploadedFile.name}</p>
                    <p className="file-size">{formatFileSize(uploadedFile.size)}</p>
                  </div>
                </div>
                <button className="remove-btn" onClick={handleRemoveFile}>
                  ✕ Remove
                </button>
              </div>
            </div>
          )}

          {/* Upload Error */}
          {uploadError && (
            <div className="upload-error">{uploadError}</div>
          )}

          {/* ── Settings ── */}
          <div className="settings">
            <div>
              <label>NUMBER OF SHORTS</label>
              <select
                value={clipCount}
                onChange={(e) => setClipCount(Number(e.target.value))}
                disabled={isLoading}
              >
                <option value={3}>3 Shorts</option>
                <option value={5}>5 Shorts</option>
                <option value={8}>8 Shorts</option>
                <option value={10}>10 Shorts</option>
                <option value={15}>15 Shorts</option>
                <option value={20}>20 Shorts</option>
                <option value={30}>30 Shorts</option>
              </select>
            </div>

            <div>
              <label>CLIP DURATION</label>
              <select defaultValue="20-30" disabled={isLoading}>
                <option value="20-30">20–30 Seconds</option>
              </select>
            </div>
          </div>

          {/* ── Required Permission Checkbox ── */}
          <div className="permission-box">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={hasPermission}
                onChange={(e) => setHasPermission(e.target.checked)}
                disabled={isLoading}
              />
              <span>
                I confirm that I own this video or have permission to download and process it.
              </span>
            </label>
          </div>

          <button
            className={`generate-btn ${isLoading ? "loading" : ""}`}
            onClick={handleGenerate}
            disabled={isLoading || (!uploadedFile && !hasPermission)}
          >
            {stage === "validating"
              ? "⏳ Validating URL..."
              : stage === "downloading"
              ? "⏳ Downloading Video..."
              : stage === "analyzing"
              ? "⏳ Analyzing Video..."
              : stage === "processing"
              ? `⏳ Generating Shorts (${progress}%)...`
              : "✨ Generate Shorts"}
          </button>

          {/* ── Live Progress Tracking ── */}
          {isLoading && (
            <div className="progress-card">
              <div className="progress-header">
                <span className="progress-title">
                  {stage === "downloading"
                    ? "📥 Downloading Video from YouTube..."
                    : stage === "analyzing"
                    ? "🔍 Analyzing Video Metadata..."
                    : `🎬 Rendering Clip ${currentClip} of ${totalClips}`}
                </span>
                <span className="progress-percent">{progress}%</span>
              </div>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width:
                      stage === "downloading"
                        ? "15%"
                        : stage === "analyzing"
                        ? "30%"
                        : `${progress}%`,
                  }}
                />
              </div>
              {activeTaskId && (
                <div className="task-id-tag">Task ID: {activeTaskId}</div>
              )}
            </div>
          )}

          {status && (
            <div
              className={`status ${
                status.startsWith("✅") || status.startsWith("🎉")
                  ? "success"
                  : status.startsWith("❌") || status.startsWith("⚠️")
                  ? "error"
                  : ""
              }`}
            >
              {status}
            </div>
          )}
        </section>

        {/* ── Generated Shorts Showcase ── */}
        {generatedClips.length > 0 && (
          <section className="shorts-showcase">
            <div className="shorts-header">
              <h2>🎉 Your Shorts Are Ready!</h2>
              <p className="subtext">
                {generatedClips.length} vertical 9:16 clips created. Preview or download below.
              </p>
            </div>
            <div className="shorts-grid">
              {generatedClips.map((clip, idx) => {
                const encodedFilename = encodeURIComponent(clip.filename);
                const clipUrl = `${BACKEND_URL}/clips/${encodedFilename}`;
                const downloadUrl = `${BACKEND_URL}/clips/${encodedFilename}?download=true`;
                const hasError = videoErrors[clip.filename];

                return (
                  <div key={clip.filename || idx} className="short-card">
                    {hasError ? (
                      <div className="video-error-placeholder">
                        <span className="error-icon">⚠️</span>
                        <p className="error-title">Preview Unavailable</p>
                        <p className="error-desc">
                          Could not stream video preview inline. You can still download the clip below.
                        </p>
                      </div>
                    ) : (
                      <video
                        className="short-video"
                        src={clipUrl}
                        controls
                        playsInline
                        preload="metadata"
                        onError={() =>
                          setVideoErrors((prev) => ({ ...prev, [clip.filename]: true }))
                        }
                      />
                    )}
                    <div className="short-meta">
                      <p className="short-title">Short #{idx + 1}</p>
                      <p className="short-details">
                        {clip.duration}s · {formatFileSize(clip.file_size)}
                      </p>
                    </div>
                    <a
                      href={downloadUrl}
                      download={clip.filename}
                      className="download-clip-btn"
                    >
                      ⬇️ Download Short
                    </a>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        <section className="features">
          <div className="feature-card">
            <span>✦</span>
            <h3>AI Highlights</h3>
            <p>Find interesting moments from long videos.</p>
          </div>

          <div className="feature-card">
            <span>▤</span>
            <h3>Auto Captions</h3>
            <p>Generate subtitles for your short videos.</p>
          </div>

          <div className="feature-card">
            <span>▶</span>
            <h3>Ready to Download</h3>
            <p>Export your clips in vertical MP4 format.</p>
          </div>
        </section>

        <footer>
          ClipAI Studio · Your videos, your creativity.
        </footer>
      </main>
    </div>
  );
}

export default App;
