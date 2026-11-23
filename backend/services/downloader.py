"""
ClipAI — Downloader Service
Downloads publicly available YouTube videos using yt-dlp.

Rules:
- Only public YouTube URLs accepted (validated before calling).
- User must confirm they have permission to download.
- Max file size: 500 MB.
- Isolated temporary directory per download job to prevent Windows file locking ([WinError 32]).
- Single fragment download on Windows (concurrent_fragment_downloads = 1).
- Per-video thread-safe lock with double-checked caching to prevent duplicate downloads.
- Reuses verified, complete, non-empty videos without re-downloading.
- Never overwrites or deletes existing valid completed videos.
- No cookies, no passwords, no login details requested, no DRM bypass.
"""

import os
import re
import time
import uuid
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DOWNLOADS_DIR = Path(__file__).parent.parent / "downloads"
MAX_FILESIZE_BYTES = 500 * 1024 * 1024   # 500 MB hard limit
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Thread-Safe Per-Video Locks ───────────────────────────────────────────────
_video_locks_guard = threading.Lock()
_video_locks: Dict[str, threading.Lock] = {}


def _get_video_lock(video_id: str) -> threading.Lock:
    """Returns a dedicated thread-safe lock for a specific video ID."""
    with _video_locks_guard:
        if video_id not in _video_locks:
            _video_locks[video_id] = threading.Lock()
        return _video_locks[video_id]


# ── FFmpeg availability check ─────────────────────────────────────────────────
def check_ffmpeg() -> bool:
    """Returns True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


# ── Safe filename helper ───────────────────────────────────────────────────────
def safe_filename(title: str, video_id: str) -> str:
    """
    Builds a filesystem-safe filename from the video title + unique video_id.
    - Removes special/unsafe characters.
    - Collapses whitespace to underscores.
    - Caps at 60 chars before the ID suffix.
    - Guarantees uniqueness via video_id.
    """
    clean = re.sub(r"[^\w\s-]", "", title or "video").strip()
    clean = re.sub(r"\s+", "_", clean)
    clean = clean.replace("..", "").strip("._-")
    if not clean:
        clean = "video"
    return f"{clean[:60]}_{video_id}.mp4"


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class DownloadResult:
    success: bool
    filename: Optional[str] = None
    title: Optional[str] = None
    file_size: Optional[int] = None   # bytes
    reason: Optional[str] = None


# ── Video Integrity & Cache Verification ──────────────────────────────────────
def _is_valid_video_file(filepath: Path) -> bool:
    """
    Verifies that a video file exists, is non-empty, and has a readable stream.
    """
    if not filepath.exists() or not filepath.is_file():
        return False

    try:
        size = filepath.stat().st_size
        if size <= 0:
            return False
    except OSError:
        return False

    probe = shutil.which("ffprobe")
    if probe:
        try:
            res = subprocess.run(
                [
                    probe,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(filepath),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0:
                dur = float(res.stdout.strip())
                return dur > 0
            return False
        except Exception:
            return False

    # Fallback if ffprobe is not installed
    return size > 1024 * 1024


def find_completed_video(video_id: str) -> Optional[Path]:
    """
    Searches DOWNLOADS_DIR for any valid, complete, non-empty video matching video_id.
    Never checks temporary directories.
    """
    if not video_id:
        return None

    # Matches *_{video_id}.mp4 or {video_id}.mp4
    candidates = list(DOWNLOADS_DIR.glob(f"*_{video_id}.mp4")) + list(DOWNLOADS_DIR.glob(f"{video_id}.mp4"))
    for candidate in candidates:
        if candidate.is_file() and not candidate.name.startswith("_temp_"):
            if _is_valid_video_file(candidate):
                return candidate
    return None


# ── Video ID Extractor helper ─────────────────────────────────────────────────
def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts an 11-character YouTube video ID directly from the URL string.
    Works for watch?v=, youtu.be/, shorts/, embed/, etc.
    """
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([\w-]{11})",
        r"^([\w-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ── Player Client & JS Runtime Configuration ──────────────────────────────────
def _get_js_runtimes() -> Dict[str, Any]:
    """
    Detects installed JavaScript runtimes (Node.js, Deno) on the local system.
    yt-dlp uses JS runtimes to solve YouTube JavaScript challenges locally without
    needing user accounts or browser cookies.
    """
    runtimes: Dict[str, Any] = {}
    if shutil.which("node"):
        runtimes["node"] = {}
    if shutil.which("deno"):
        runtimes["deno"] = {}
    return runtimes or {"deno": {}}


def _get_base_ydl_opts(extra_extractor_clients: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Builds the base yt-dlp configuration for reliable public YouTube extraction:
    - Forces IPv4 (0.0.0.0) to avoid dynamic/flagged IPv6 ranges that trigger bot checks.
    - Enables local JS runtime (Node.js v22+) for solving client challenges.
    - Specifies supported player client fallbacks.
    """
    clients = extra_extractor_clients or ["tv_samsung", "visionos", "web", "android", "ios", "mweb"]
    return {
        "source_address": "0.0.0.0",  # Forces IPv4
        "js_runtimes": _get_js_runtimes(),
        "extractor_args": {
            "youtube": {
                "player_client": clients,
            }
        },
    }


# ── Granular Error Classification ─────────────────────────────────────────────
def classify_youtube_error(exc: Exception) -> str:
    """
    Classifies yt-dlp exceptions into user-friendly, specific error messages.
    Does NOT misclassify bot challenges, rate-limits, or network issues as private videos.
    """
    error_msg = str(exc)
    error_lower = error_msg.lower()

    # 1. Bot verification / CAPTCHA
    if (
        "sign in to confirm you're not a bot" in error_lower
        or "sign in to confirm you’re not a bot" in error_lower
        or "confirm you're not a bot" in error_lower
        or "confirm you’re not a bot" in error_lower
        or "playercaptchaviewmodel" in error_lower
        or "captcha challenge" in error_lower
    ):
        return (
            "YouTube is currently requesting bot verification for this video. "
            "Please wait a few moments and try again."
        )

    # 2. Private video
    if (
        "private video" in error_lower
        or "this video is private" in error_lower
        or "video is private" in error_lower
    ):
        return "This video is private. Only publicly accessible videos can be processed."

    # 3. Login required / Age-restricted / Members-only
    if (
        "sign in to confirm your age" in error_lower
        or "age-restricted" in error_lower
        or "members-only" in error_lower
        or "join this channel" in error_lower
        or "requires payment" in error_lower
        or "sign in" in error_lower
    ):
        return (
            "This video requires signing in (e.g. age-restricted or members-only). "
            "Only publicly accessible videos can be processed."
        )

    # 4. Geo-restricted
    if (
        "not made this video available in your country" in error_lower
        or "not available in your country" in error_lower
        or "geo_restricted" in error_lower
        or "geo-restricted" in error_lower
        or "georestricted" in error_lower
    ):
        return "This video is not available in your region due to geographic restrictions."

    # 5. Unavailable / Removed / Terminated / Deleted
    if (
        "video unavailable" in error_lower
        or "this video is not available" in error_lower
        or "has been removed" in error_lower
        or "account associated with this video has been terminated" in error_lower
        or "copyright claim" in error_lower
        or "deleted" in error_lower
        or "unavailable" in error_lower
    ):
        return "This video is unavailable, deleted, or removed."

    # 6. Rate-limited / HTTP 429
    if (
        "http error 429" in error_lower
        or "too many requests" in error_lower
        or "rate-limited" in error_lower
        or "this content isn't available, try again later" in error_lower
    ):
        return "YouTube request rate limit reached. Please wait a few moments and try again."

    # 7. File size limit
    if (
        "file is larger than max-filesize" in error_lower
        or "larger than max-filesize" in error_lower
    ):
        return f"Video exceeds the {MAX_FILESIZE_BYTES // (1024 * 1024)} MB size limit."

    # 8. Network connection / DNS / Timeout errors
    if (
        "getaddrinfo failed" in error_lower
        or "winerror 10060" in error_lower
        or "timed out" in error_lower
        or "timeout" in error_lower
        or "connection refused" in error_lower
        or "connection reset" in error_lower
        or "remotedisconnected" in error_lower
        or "network is unreachable" in error_lower
        or "failed to establish a new connection" in error_lower
    ):
        return "Network error connecting to YouTube. Please check your internet connection."

    # 9. Generic clean fallback
    clean_err = re.sub(r"[^\w\s:.-]", "", error_msg)[:120].strip()
    return f"Download failed: {clean_err}" if clean_err else "Download failed due to an unexpected error."


# ── Core download function ─────────────────────────────────────────────────────
def download_video(url: str) -> DownloadResult:
    """
    Downloads a public YouTube video without requiring cookies or logins:
    - Checks cache before download.
    - Uses per-video thread-safe lock with double-checked re-check.
    - Runs download in an isolated unique temporary directory per job.
    - Single fragment download on Windows to prevent WinError 32.
    - Never overwrites or deletes existing completed videos.
    - Safe retry handling and granular error classification.
    """
    try:
        import yt_dlp
    except ImportError:
        return DownloadResult(success=False, reason="yt-dlp is not installed.")

    # Step 0: Fast URL-based cache check (works even without network call)
    fast_video_id = extract_video_id(url)
    if fast_video_id:
        existing = find_completed_video(fast_video_id)
        if existing:
            logger.info("Found existing valid video from URL parse: %s", existing.name)
            return DownloadResult(
                success=True,
                filename=existing.name,
                title=existing.stem,
                file_size=existing.stat().st_size,
            )

    ffmpeg_ok = check_ffmpeg()

    # Step 1: Extract info without downloading (fast metadata check with safe retry)
    base_opts = _get_base_ydl_opts()
    ydl_info_opts = {
        **base_opts,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,
    }

    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        logger.warning("First metadata extraction attempt failed: %s. Retrying with fallback player clients...", exc)
        fallback_opts = {
            **_get_base_ydl_opts(extra_extractor_clients=["mweb", "web_embedded", "android", "ios", "tv"]),
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "noplaylist": True,
        }
        try:
            time.sleep(1.0)
            with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as retry_exc:
            reason = classify_youtube_error(retry_exc)
            logger.error("Metadata extraction failed after retry: %s", retry_exc)
            return DownloadResult(success=False, reason=reason)

    if info is None:
        return DownloadResult(
            success=False,
            reason="Could not retrieve video information. The video may be private, deleted, or unavailable."
        )

    video_id = info.get("id") or fast_video_id or "unknown"
    title = info.get("title", "Untitled")
    filesize = info.get("filesize") or info.get("filesize_approx")

    # Pre-flight size check
    if filesize and filesize > MAX_FILESIZE_BYTES:
        return DownloadResult(
            success=False,
            reason=f"Video is too large ({filesize // (1024*1024)} MB). Maximum allowed size is {MAX_FILESIZE_BYTES // (1024*1024)} MB."
        )

    # ── Check 1: Before acquiring lock (with extracted info title) ─────────────
    existing = find_completed_video(video_id)
    if existing:
        logger.info("Found existing valid video for ID %s: %s", video_id, existing.name)
        return DownloadResult(
            success=True,
            filename=existing.name,
            title=title,
            file_size=existing.stat().st_size,
        )

    # ── Acquire per-video lock (concurrent requests for same video will wait) ──
    lock = _get_video_lock(video_id)
    with lock:
        # ── Check 2: Inside lock (in case concurrent thread finished it) ───────
        existing_after_lock = find_completed_video(video_id)
        if existing_after_lock:
            logger.info("Concurrent download finished; reusing valid video: %s", existing_after_lock.name)
            return DownloadResult(
                success=True,
                filename=existing_after_lock.name,
                title=title,
                file_size=existing_after_lock.stat().st_size,
            )

        # ── Setup isolated unique temp directory for this download job ─────────
        job_id = uuid.uuid4().hex[:8]
        job_temp_dir = DOWNLOADS_DIR / f"_temp_{video_id}_{job_id}"
        job_temp_dir.mkdir(parents=True, exist_ok=True)

        if ffmpeg_ok:
            fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        else:
            fmt = "best[ext=mp4]/best"

        ydl_opts = {
            **base_opts,
            "format": fmt,
            "paths": {
                "home": str(job_temp_dir),
                "temp": str(job_temp_dir),
            },
            "outtmpl": {"default": "%(id)s.%(ext)s"},
            "concurrent_fragment_downloads": 1,  # Windows-safe single fragment download
            "windowsfilenames": True,
            "max_filesize": MAX_FILESIZE_BYTES,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": False,
            "ignoreerrors": False,
            "nocheckcertificate": False,
            "geo_bypass": False,
            **({"merge_output_format": "mp4"} if ffmpeg_ok else {}),
        }

        try:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as exc:
                logger.warning("Download failed on primary attempt: %s. Retrying with fallback clients...", exc)
                retry_opts = {
                    **ydl_opts,
                    **_get_base_ydl_opts(extra_extractor_clients=["mweb", "web_embedded", "android", "ios", "tv"]),
                }
                time.sleep(1.0)
                with yt_dlp.YoutubeDL(retry_opts) as ydl:
                    ydl.download([url])

            # Find downloaded file inside job_temp_dir
            matches = list(job_temp_dir.glob(f"{video_id}.*"))
            if not matches:
                matches = [
                    f for f in job_temp_dir.iterdir()
                    if f.is_file() and not f.name.endswith(".part") and not f.name.endswith(".ytdl")
                ]

            if not matches:
                return DownloadResult(
                    success=False,
                    reason="Download completed but file not found in temporary directory."
                )

            temp_file = matches[0]
            final_name = safe_filename(title, video_id)
            final_path = DOWNLOADS_DIR / final_name

            # Condition 2: Never delete or overwrite the only valid completed video
            if final_path.exists():
                if _is_valid_video_file(final_path):
                    logger.info("Final path '%s' already has a valid file; preserving it.", final_name)
                    temp_file.unlink(missing_ok=True)
                    return DownloadResult(
                        success=True,
                        filename=final_name,
                        title=title,
                        file_size=final_path.stat().st_size,
                    )

            # Safely move temp_file to final destination
            shutil.move(str(temp_file), str(final_path))

            # Verify integrity of final moved file
            if not _is_valid_video_file(final_path):
                return DownloadResult(
                    success=False,
                    reason="Downloaded video could not be verified as a valid, readable MP4 file."
                )

            actual_size = final_path.stat().st_size
            logger.info("Successfully downloaded and verified: %s (%d MB)", final_name, actual_size // (1024 * 1024))

            return DownloadResult(
                success=True,
                filename=final_name,
                title=title,
                file_size=actual_size,
            )

        except Exception as exc:
            reason = classify_youtube_error(exc)
            logger.error("Download error for %s: %s (classified: %s)", url, exc, reason)
            return DownloadResult(success=False, reason=reason)

        finally:
            # Clean up temporary folder only after completion or safe failure
            try:
                if job_temp_dir.exists():
                    shutil.rmtree(job_temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning("Could not remove temp dir %s: %s", job_temp_dir, e)
