"""
ClipAI — FastAPI Backend
Phase 4: health check + URL validation + downloading + clip cutting.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path

from validators.url_validator import validate_url
from services.downloader import download_video, check_ffmpeg
from services.clipper import (
    process_video,
    PROCESSED_DIR,
    DOWNLOADS_DIR,
    get_video_details,
    validate_clip_request,
)
from services.task_manager import task_manager

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClipAI Backend",
    description="Backend API for ClipAI — AI Shorts Maker",
    version="0.4.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class UrlRequest(BaseModel):
    url: str


class UrlResponse(BaseModel):
    valid: bool
    platform: Optional[str]
    url: str
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    ffmpeg: str


class DownloadRequest(BaseModel):
    url: str
    confirmed_permission: bool


class DownloadResponse(BaseModel):
    status: str
    filename: Optional[str] = None
    title: Optional[str] = None
    file_size: Optional[int] = None
    reason: Optional[str] = None


class ProcessRequest(BaseModel):
    filename: str = Field(
        ...,
        description="Filename of the downloaded video (from POST /download response).",
        examples=["My_Video_dQw4w9WgXcQ.mp4"],
    )
    clip_count: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Number of clips to generate (1–30).",
    )
    clip_duration: int = Field(
        default=25,
        ge=20,
        le=30,
        description="Duration of each clip in seconds (20–30).",
    )
    detection_mode: str = Field(
        default="ai",
        description="Highlight detection mode: 'ai' (default) or 'even'.",
    )
    auto_captions: bool = Field(
        default=True,
        description="Whether to automatically generate and burn subtitles into the clips (default: True).",
    )


class ClipInfo(BaseModel):
    success: bool
    filename: Optional[str] = None
    duration: Optional[int] = None    # seconds
    file_size: Optional[int] = None   # bytes
    reason: Optional[str] = None


class ProcessResponse(BaseModel):
    status: str                        # "success" | "partial" | "error"
    clips: List[ClipInfo] = []
    reason: Optional[str] = None


class ProcessTaskResponse(BaseModel):
    task_id: str
    status: str                        # "queued"
    message: str
    filename: str
    clip_count: int
    clip_duration: int
    detection_mode: str = "ai"
    auto_captions: bool = True


class ProcessStatusResponse(BaseModel):
    task_id: str
    filename: str
    status: str                        # "queued" | "analyzing_highlights" | "processing" | "completed" | "failed"
    progress: int                      # 0 to 100 percentage
    current_clip: int
    total_clips: int
    completed_clips: int
    clips: List[ClipInfo] = []
    error: Optional[str] = None
    created_at: str
    updated_at: str


class VideoInfoResponse(BaseModel):
    filename: str
    is_supported: bool
    message: str
    duration: float
    duration_formatted: str
    width: int
    height: int
    recommended_clip_count: int
    min_clip_count: int
    max_clip_count: int


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns backend status and FFmpeg availability."""
    ffmpeg_status = "available" if check_ffmpeg() else "not found - clip processing disabled"
    return HealthResponse(
        status="ok",
        message="ClipAI backend is running.",
        ffmpeg=ffmpeg_status,
    )


@app.post("/validate-url", response_model=UrlResponse, tags=["URL"])
def validate_video_url(body: UrlRequest):
    """Validates a YouTube or Instagram URL without accessing content."""
    if not body.url or not body.url.strip():
        raise HTTPException(status_code=422, detail="URL field is required.")
    result = validate_url(body.url)
    return UrlResponse(
        valid=result.valid,
        platform=result.platform,
        url=result.url,
        reason=result.reason,
    )


@app.post("/download", response_model=DownloadResponse, tags=["Download"])
def download_video_endpoint(body: DownloadRequest):
    """
    Downloads a publicly available YouTube video.
    - confirmed_permission must be true.
    - Max 500 MB. No login, no DRM bypass.
    """
    if not body.confirmed_permission:
        raise HTTPException(
            status_code=403,
            detail="You must confirm you have permission to download this video."
        )
    if not body.url or not body.url.strip():
        raise HTTPException(status_code=422, detail="URL field is required.")

    validation = validate_url(body.url)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.reason or "Invalid URL.")
    if validation.platform != "youtube":
        raise HTTPException(
            status_code=422,
            detail="Only YouTube video downloading is supported at this time."
        )

    result = download_video(validation.url)
    if not result.success:
        return DownloadResponse(status="error", reason=result.reason)

    return DownloadResponse(
        status="success",
        filename=result.filename,
        title=result.title,
        file_size=result.file_size,
    )


@app.post("/process", response_model=ProcessTaskResponse, tags=["Process"])
def process_video_endpoint(body: ProcessRequest):
    """
    Submits a video clipping job to the background queue and returns immediately.
    - Pre-flight validates file existence, duration, and non-overlapping timestamps.
    - Rejects invalid requests immediately with HTTP 422 before queuing.
    - Sequential queue: Only ONE FFmpeg job runs at a time; others wait in "queued" status.
    - Use GET /process-status/{task_id} to track live progress.
    """
    safe_name = Path(body.filename).name
    if safe_name != body.filename:
        raise HTTPException(
            status_code=422,
            detail="Invalid filename. Do not include directory paths."
        )

    # Pre-flight validation (file, duration, non-overlapping timestamps)
    is_valid, err_reason = validate_clip_request(
        filename=safe_name,
        clip_count=body.clip_count,
        clip_duration=body.clip_duration,
    )
    if not is_valid:
        raise HTTPException(status_code=422, detail=err_reason)

    if body.detection_mode not in ("ai", "even"):
        raise HTTPException(
            status_code=422,
            detail="Invalid detection_mode. Must be 'ai' or 'even'."
        )

    # Submit task to single-worker in-memory background queue
    task_id = task_manager.submit_task(
        filename=safe_name,
        clip_count=body.clip_count,
        clip_duration=body.clip_duration,
        processor_fn=process_video,
        detection_mode=body.detection_mode,
        auto_captions=body.auto_captions,
    )

    return ProcessTaskResponse(
        task_id=task_id,
        status="queued",
        message="Processing job queued. Only one FFmpeg job runs at a time. Track progress via GET /process-status/{task_id}.",
        filename=safe_name,
        clip_count=body.clip_count,
        clip_duration=body.clip_duration,
        detection_mode=body.detection_mode,
        auto_captions=body.auto_captions,
    )


@app.get("/process-status/{task_id}", response_model=ProcessStatusResponse, tags=["Process"])
def get_process_status_endpoint(task_id: str):
    """
    Returns live progress and results for a background video processing job.
    NOTE: All task states are kept in memory and are lost if the backend server restarts.
    """
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Either it never existed or the backend was restarted."
        )

    return ProcessStatusResponse(
        task_id=task["task_id"],
        filename=task["filename"],
        status=task["status"],
        progress=task["progress"],
        current_clip=task["current_clip"],
        total_clips=task["total_clips"],
        completed_clips=task["completed_clips"],
        clips=[ClipInfo(**c) for c in task["clips"]],
        error=task["error"],
        created_at=task["created_at"],
        updated_at=task["updated_at"],
    )


@app.get("/clips/{filename}", tags=["Process"])
def download_clip(filename: str, download: bool = False):
    """
    Serves a processed clip as an MP4 file.

    - Default (download=False): Streams the video inline with 'Content-Disposition: inline'
      and 'Accept-Ranges: bytes' for browser <video> player preview and duration detection.
    - download=True: Returns 'Content-Disposition: attachment' for direct file downloading.
    """
    # Reject path traversal
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=422, detail="Invalid filename.")

    clip_path = PROCESSED_DIR / safe_name
    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Clip '{safe_name}' not found. Run POST /process first."
        )

    disposition_type = "attachment" if download else "inline"

    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=safe_name,
        content_disposition_type=disposition_type,
    )


@app.get("/video-info/{filename}", response_model=VideoInfoResponse, tags=["Video Info"])
def get_video_info_endpoint(filename: str):
    """
    Inspects a video from backend/downloads/ using FFprobe.
    Returns duration, resolution, and non-overlapping clip recommendations.
    """
    # Reject path traversal
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=422, detail="Invalid filename format.")

    filepath = DOWNLOADS_DIR / safe_name
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Video '{safe_name}' not found in downloads/ folder."
        )

    try:
        info = get_video_details(filepath)
        return VideoInfoResponse(**info)
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read video metadata: {str(e)}"
        )



