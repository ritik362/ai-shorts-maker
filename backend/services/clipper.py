"""
ClipAI — Clipper Service  (Phase 4)
Cuts a downloaded video into short vertical clips using FFmpeg.

TEMPORARY LOGIC NOTE:
  Clip start points are selected via EVEN SPACING across the video duration.
  e.g., a 10-min video split into 3 clips → starts at 0s, 200s, 400s.
  This will be replaced by AI-based highlight detection in a future phase.

Output format: 9:16 vertical (1080×1920) with blurred-background framing.
  - Foreground: original video scaled to fit within 1080×1920 (no crop).
  - Background: same video scaled/cropped to fill 1080×1920, then blurred.
  - This avoids cutting off subjects that may not be centered.

Constraints:
  - Sequential processing (RAM-friendly for 8 GB Windows PC).
  - Original source file is never modified or deleted.
  - Partial output files are cleaned up on any error.
"""

import re
import json
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)

DOWNLOADS_DIR = Path(__file__).parent.parent / "downloads"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Output dimensions for vertical Shorts / Reels format
OUT_WIDTH  = 1080
OUT_HEIGHT = 1920

# Encoding settings — balanced for quality vs. speed on 8 GB RAM
VIDEO_CODEC  = "libx264"
AUDIO_CODEC  = "aac"
CRF          = "23"          # 18 = near-lossless, 28 = smaller; 23 is default
PRESET       = "fast"        # veryfast/fast/medium — fast is good for 8 GB


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    total_sec = max(0, int(round(seconds)))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def calculate_clip_recommendations(duration: float, clip_duration: int = 25) -> Dict[str, Any]:
    """
    Calculates dynamic min, max, and recommended clip counts based on video duration.
    - Hard limit of 30 clips.
    - Verifies non-overlapping timestamps.
    - Matches duration tier table.
    """
    if duration < 20.0:
        return {
            "is_supported": False,
            "message": f"Video is too short ({int(duration)}s). Minimum 20 seconds required for shorts.",
            "min_clip_count": 0,
            "max_clip_count": 0,
            "recommended_clip_count": 0,
        }

    # Dynamic maximum non-overlapping clips capped at 30
    dynamic_max = int(duration // clip_duration)
    max_clips = min(30, max(1, dynamic_max))

    # Duration-based recommendation tiers
    if duration < 60:
        # 20s - 60s
        recommended = 1
    elif duration < 300:
        # 1 - 5 mins (60s - 300s): 3 - 5 clips
        recommended = 3 + int(2 * ((duration - 60) / 240))
    elif duration < 900:
        # 5 - 15 mins (300s - 900s): 8 - 10 clips
        recommended = 8 + int(2 * ((duration - 300) / 600))
    elif duration < 2700:
        # 15 - 45 mins (900s - 2700s): 15 - 20 clips
        recommended = 15 + int(5 * ((duration - 900) / 1800))
    else:
        # 45 mins to 2+ hours: 25 - 30 clips (smooth scaling, not forcing 30)
        progress = min(1.0, (duration - 2700) / 4500)
        recommended = 25 + int(5 * progress)

    # Cap recommended at max_clips and hard cap 30
    recommended = min(max_clips, recommended)
    recommended = max(1, recommended)

    # Strict timestamp verification to guarantee non-overlapping intervals
    while recommended > 1:
        step = (duration - clip_duration) / (recommended - 1)
        if step >= clip_duration:
            break
        recommended -= 1

    return {
        "is_supported": True,
        "message": "Video is ready for clipping.",
        "min_clip_count": 1,
        "max_clip_count": max_clips,
        "recommended_clip_count": recommended,
    }


def get_video_details(filepath: Path) -> Dict[str, Any]:
    """
    Inspects video file metadata using FFprobe JSON mode.
    Reads only header metadata without loading video frames into RAM.
    """
    probe = ffprobe_path()
    if not probe:
        raise RuntimeError("FFprobe is not installed or not found on PATH.")

    try:
        result = subprocess.run(
            [
                probe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise ValueError(f"FFprobe execution failed: {result.stderr.strip()}")

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        format_info = data.get("format", {})

        # Extract duration from format first, fallback to video stream
        raw_duration = format_info.get("duration")
        if not raw_duration and streams:
            raw_duration = streams[0].get("duration")

        if not raw_duration:
            raise ValueError("Could not determine video duration from file metadata.")

        duration = float(raw_duration)

        width = 0
        height = 0
        if streams:
            width = int(streams[0].get("width", 0))
            height = int(streams[0].get("height", 0))

        rec = calculate_clip_recommendations(duration)

        return {
            "filename": filepath.name,
            "is_supported": rec["is_supported"],
            "message": rec["message"],
            "duration": round(duration, 2),
            "duration_formatted": format_duration(duration),
            "width": width,
            "height": height,
            "recommended_clip_count": rec["recommended_clip_count"],
            "min_clip_count": rec["min_clip_count"],
            "max_clip_count": rec["max_clip_count"],
        }
    except Exception as exc:
        logger.error("Failed to read video details for %s: %s", filepath, exc)
        raise


def ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def ffprobe_path() -> Optional[str]:
    return shutil.which("ffprobe")


def _safe_clip_name(source_stem: str, index: int) -> str:
    """Build a filesystem-safe output filename for a clip."""
    clean = re.sub(r"[^\w-]", "_", source_stem)[:50]
    return f"clip_{index + 1}_{clean}.mp4"


def _get_video_duration(filepath: Path) -> Optional[float]:
    """
    Use ffprobe to get video duration in seconds.
    Returns None if ffprobe fails or file is unreadable.
    """
    probe = ffprobe_path()
    if not probe:
        return None
    try:
        result = subprocess.run(
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
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", filepath, exc)
        return None


def _build_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: int,
    ass_path: Optional[Path] = None,
) -> List[str]:
    """
    Build the FFmpeg command that:
      1. Seeks to `start_sec` (input-side seek — fast).
      2. Cuts `duration_sec` seconds.
      3. Applies blurred-background + scaled-foreground filter for 9:16 output.
      4. Optionally burns ASS subtitles in the SAME filtergraph pass.

    Filter graph explanation:
      [0:v] split into two streams:
        [bg]: scale to FILL 1080×1920 (may crop edges), then heavy boxblur → background
        [fg]: scale to FIT within 1080×1920 (letterbox, no crop)      → foreground
      Overlay [fg] centered on [bg].
      If ass_path provided: render subtitles on top of composite via ass filter.
    """
    bg_filter = (
        f"scale={OUT_WIDTH}:{OUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={OUT_WIDTH}:{OUT_HEIGHT},"
        f"boxblur=20:5"
    )
    fg_filter = (
        f"scale={OUT_WIDTH}:{OUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_WIDTH}:{OUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black@0"
    )

    if ass_path and ass_path.exists():
        # Escape path for FFmpeg filter on Windows (forward slashes and escaped colon)
        escaped_ass = ass_path.as_posix().replace(":", "\\:")
        filter_complex = (
            f"[0:v]split=2[raw1][raw2];"
            f"[raw1]{bg_filter}[bg];"
            f"[raw2]{fg_filter}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vcomp];"
            f"[vcomp]ass='{escaped_ass}'[out]"
        )
    else:
        filter_complex = (
            f"[0:v]split=2[raw1][raw2];"
            f"[raw1]{bg_filter}[bg];"
            f"[raw2]{fg_filter}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[out]"
        )

    return [
        ffmpeg_path(),
        "-y",                          # overwrite output without asking
        "-ss", str(start_sec),         # input-side seek (fast)
        "-t",  str(duration_sec),      # duration to cut
        "-i",  str(input_path),        # input file
        "-filter_complex", filter_complex,
        "-map", "[out]",               # video from filter graph
        "-map", "0:a?",                # audio (optional — keep if present)
        "-c:v", VIDEO_CODEC,
        "-crf", CRF,
        "-preset", PRESET,
        "-c:a", AUDIO_CODEC,
        "-movflags", "+faststart",     # web-friendly MP4 (moov atom at front)
        str(output_path),
    ]


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ClipResult:
    success: bool
    filename: Optional[str]   = None
    duration: Optional[int]   = None   # seconds
    file_size: Optional[int]  = None   # bytes
    reason: Optional[str]     = None


@dataclass
class ProcessResult:
    success: bool
    clips: List[ClipResult]   = field(default_factory=list)
    reason: Optional[str]     = None


# ── Main entry point ───────────────────────────────────────────────────────────

def validate_clip_request(
    filename: str,
    clip_count: int,
    clip_duration: int,
) -> Tuple[bool, Optional[str]]:
    """
    Pre-flight validation before queuing a processing job.
    Validates file existence, FFmpeg availability, duration, and non-overlapping timestamps.
    """
    if not ffmpeg_path():
        return False, "FFmpeg is not installed or not found on PATH."

    safe_name = Path(filename).name
    source = DOWNLOADS_DIR / safe_name
    if not source.exists() or not source.is_file():
        return False, f"Source file '{safe_name}' not found in downloads/. Download it first using POST /download."

    total_duration = _get_video_duration(source)
    if total_duration is None:
        return False, "Could not read video duration. File may be corrupt or unreadable."

    usable = total_duration - clip_duration
    if usable < 0:
        return False, f"Video duration ({int(total_duration)}s) is shorter than requested clip duration ({clip_duration}s)."

    max_non_overlapping = min(30, int(total_duration // clip_duration))
    if clip_count > 1:
        step = usable / (clip_count - 1)
        starts = [i * step for i in range(clip_count)]
        for i in range(len(starts) - 1):
            current_end = starts[i] + clip_duration
            next_start = starts[i + 1]
            if current_end > next_start + 1e-4:
                return False, (
                    f"Video duration ({int(total_duration)}s) is too short for {clip_count} non-overlapping "
                    f"clips of {clip_duration}s each. Clip {i + 1} ends at {current_end:.1f}s while clip {i + 2} "
                    f"starts at {next_start:.1f}s (overlap detected). "
                    f"Maximum non-overlapping clips allowed for this video is {max_non_overlapping}."
                )

    return True, None


def _calculate_even_starts(clip_count: int, total_duration: float, clip_duration: int) -> List[float]:
    """Helper to calculate evenly spaced start timestamps across video duration."""
    usable = max(0.0, total_duration - clip_duration)
    if clip_count == 1:
        return [round(usable / 2, 2)]
    step = usable / (clip_count - 1)
    return [round(i * step, 2) for i in range(clip_count)]


def process_video(
    filename: str,
    clip_count: int  = 3,
    clip_duration: int = 25,     # seconds per clip
    progress_callback: Optional[Callable[..., None]] = None,
    detection_mode: str = "ai",
    auto_captions: bool = True,
) -> ProcessResult:
    """
    Cut `clip_count` clips of `clip_duration` seconds from the given downloaded
    video file and convert each to vertical 9:16 (1080×1920).

    Highlight Selection:
    - detection_mode="ai": Uses faster-whisper and heuristic NLP scoring to find
      engaging dialogues, questions, and hooks. Falls back to even spacing if
      no speech is detected or an error occurs.
    - detection_mode="even": Uses strictly even spacing across video duration.

    Auto Captions:
    - auto_captions=True (default): Automatically formats, synchronizes, and burns
      styled ASS subtitles (Nirmala UI font) directly onto the video in a single pass.
      Reuses transcription results from highlight detection to avoid re-transcription.
    - auto_captions=False: Renders clean video without subtitles.

    The original source file is NEVER modified or deleted.
    Supports progress_callback for live background tracking.
    """

    # ── 0. Guard: FFmpeg present ───────────────────────────────────────────────
    if not ffmpeg_path():
        return ProcessResult(
            success=False,
            reason="FFmpeg is not installed or not found on PATH. Please install FFmpeg."
        )

    # ── 1. Resolve & validate source file ─────────────────────────────────────
    safe_name = Path(filename).name
    source = DOWNLOADS_DIR / safe_name

    if not source.exists() or not source.is_file():
        return ProcessResult(
            success=False,
            reason=f"Source file '{safe_name}' not found in downloads/. "
                   "Please download the video first using POST /download."
        )

    # ── 2. Get video duration via ffprobe ──────────────────────────────────────
    total_duration = _get_video_duration(source)
    if total_duration is None:
        return ProcessResult(
            success=False,
            reason="Could not read video duration. The file may be corrupt or unsupported."
        )

    # Calculate maximum non-overlapping clips possible
    max_non_overlapping = min(30, int(total_duration // clip_duration))

    # Reject if video is shorter than a single clip duration
    usable = total_duration - clip_duration
    if usable < 0:
        return ProcessResult(
            success=False,
            reason=(
                f"Video duration ({int(total_duration)}s) is shorter than requested "
                f"clip duration ({clip_duration}s)."
            )
        )

    # ── 3. Determine start points: AI Highlight Detection or Even Spacing ─────
    starts: List[float] = []
    cached_speech_segments: List[Any] = []

    if detection_mode == "ai":
        if progress_callback:
            progress_callback(stage="analyzing_highlights")
        logger.info("Invoking AI Highlight Detection for '%s'...", safe_name)
        try:
            from services.highlight_detector import detect_highlights

            detection_res = detect_highlights(
                video_path=source,
                total_duration=total_duration,
                clip_count=clip_count,
                clip_duration=clip_duration,
            )

            # Save speech segments for subtitle reuse (saves CPU and model reload)
            if detection_res.speech_segments:
                cached_speech_segments = detection_res.speech_segments

            if detection_res.success and len(detection_res.candidates) >= clip_count:
                starts = [round(c.start, 2) for c in detection_res.candidates[:clip_count]]
                logger.info(
                    "AI Highlight Detection succeeded. Selected %d candidates: %s",
                    len(starts), [f"{s:.1f}s" for s in starts]
                )
            elif detection_res.success and len(detection_res.candidates) > 0:
                logger.info(
                    "AI detected %d candidates (requested %d). Falling back to even spacing to ensure requested clip count.",
                    len(detection_res.candidates), clip_count
                )
                starts = _calculate_even_starts(clip_count, total_duration, clip_duration)
            else:
                logger.info(
                    "AI Highlight Detection returned fallback (reason: %s). Using even spacing.",
                    detection_res.reason or "No speech detected"
                )
                starts = _calculate_even_starts(clip_count, total_duration, clip_duration)
        except Exception as det_exc:
            logger.warning(
                "AI Highlight Detection encountered an error: %s. Falling back to even spacing.",
                det_exc, exc_info=True
            )
            starts = _calculate_even_starts(clip_count, total_duration, clip_duration)
    else:
        logger.info("detection_mode='even' requested. Using evenly spaced start points.")
        starts = _calculate_even_starts(clip_count, total_duration, clip_duration)

    # Sort start timestamps chronologically for natural video viewing
    starts.sort()

    # Explicit timestamp interval validation to GUARANTEE non-overlapping intervals
    for i in range(len(starts) - 1):
        current_end = starts[i] + clip_duration
        next_start = starts[i + 1]
        if current_end > next_start + 1e-4:
            logger.warning(
                "Overlap detected in candidate starts (%s). Falling back to guaranteed even spacing.", starts
            )
            starts = _calculate_even_starts(clip_count, total_duration, clip_duration)
            starts.sort()
            break

    logger.info(
        "Starting sequential rendering of %d clips of %ds from '%s' (mode=%s, auto_captions=%s, %.1fs total). "
        "Start points: %s",
        clip_count, clip_duration, safe_name, detection_mode, auto_captions, total_duration,
        [f"{s:.1f}s" for s in starts],
    )

    # ── 4. Cut clips sequentially (RAM and CPU friendly) ───────────────────────
    source_stem  = source.stem
    clip_results: List[ClipResult] = []

    for idx, start in enumerate(starts):
        out_name = _safe_clip_name(source_stem, idx)
        out_path = PROCESSED_DIR / out_name

        logger.info(
            "Rendering clip %d/%d: '%s' (from %.1fs to %.1fs)...",
            idx + 1, clip_count, out_name, start, start + clip_duration
        )

        if progress_callback:
            progress_callback(
                current_clip=idx + 1,
                total_clips=clip_count,
                completed_clip_info=None,
                stage="processing",
            )

        # Generate subtitles if auto_captions is enabled
        ass_path = None
        if auto_captions:
            try:
                from services.caption_generator import (
                    generate_clip_captions_from_segments,
                    transcribe_and_generate_clip_captions,
                )
                temp_ass_path = PROCESSED_DIR / f"_temp_sub_{source_stem[:15]}_{idx + 1}_{uuid.uuid4().hex[:6]}.ass"

                if cached_speech_segments:
                    # Reuses existing speech segments from highlight detection (0 extra AI seconds)
                    ass_path = generate_clip_captions_from_segments(
                        speech_segments=cached_speech_segments,
                        clip_start=start,
                        clip_duration=clip_duration,
                        output_ass_path=temp_ass_path,
                    )
                else:
                    # On-demand transcription of this 25s clip only (~1.5s on CPU)
                    ass_path = transcribe_and_generate_clip_captions(
                        video_path=source,
                        clip_start=start,
                        clip_duration=clip_duration,
                        output_ass_path=temp_ass_path,
                    )
            except Exception as cap_err:
                logger.warning("Could not generate captions for clip %d: %s", idx + 1, cap_err)
                ass_path = None

        cmd = _build_ffmpeg_cmd(source, out_path, round(start, 2), clip_duration, ass_path=ass_path)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,          # 5-min timeout per clip
            )

            if proc.returncode != 0:
                # Capture FFmpeg stderr for debugging
                stderr_tail = proc.stderr[-500:] if proc.stderr else "(no output)"
                logger.error("FFmpeg failed for clip %d: %s", idx + 1, stderr_tail)
                _cleanup_file(out_path)
                err_reason = f"FFmpeg error on clip {idx + 1}. Check server logs."
                if progress_callback:
                    progress_callback(
                        current_clip=idx + 1,
                        total_clips=clip_count,
                        completed_clip_info=None,
                        is_failed=True,
                        err_msg=err_reason,
                    )
                return ProcessResult(success=False, clips=clip_results, reason=err_reason)
            else:
                size = out_path.stat().st_size if out_path.exists() else 0
                logger.info("Clip %d done: %s (%d MB)", idx + 1, out_name, size // (1024 * 1024))
                clip_info = ClipResult(
                    success=True,
                    filename=out_name,
                    duration=clip_duration,
                    file_size=size,
                )
                clip_results.append(clip_info)
                if progress_callback:
                    progress_callback(
                        current_clip=idx + 1,
                        total_clips=clip_count,
                        completed_clip_info={
                            "success": True,
                            "filename": out_name,
                            "duration": clip_duration,
                            "file_size": size,
                        },
                        stage="processing",
                    )

        except subprocess.TimeoutExpired:
            _cleanup_file(out_path)
            err_reason = f"Clip {idx + 1} timed out after 5 minutes."
            if progress_callback:
                progress_callback(
                    current_clip=idx + 1,
                    total_clips=clip_count,
                    completed_clip_info=None,
                    is_failed=True,
                    err_msg=err_reason,
                )
            return ProcessResult(success=False, clips=clip_results, reason=err_reason)
        except Exception as exc:
            _cleanup_file(out_path)
            err_reason = f"Unexpected error on clip {idx + 1}: {exc}"
            if progress_callback:
                progress_callback(
                    current_clip=idx + 1,
                    total_clips=clip_count,
                    completed_clip_info=None,
                    is_failed=True,
                    err_msg=err_reason,
                )
            return ProcessResult(success=False, clips=clip_results, reason=err_reason)
        finally:
            # Clean up temporary ASS subtitle file immediately after clip finishes
            if ass_path and ass_path.exists():
                try:
                    ass_path.unlink()
                except Exception:
                    pass

    # ── 5. Determine overall success ───────────────────────────────────────────
    all_success = (len(clip_results) == clip_count) and all(c.success for c in clip_results)
    return ProcessResult(success=all_success, clips=clip_results)


def _cleanup_file(path: Path) -> None:
    """Remove an incomplete or failed output file, if it exists."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
