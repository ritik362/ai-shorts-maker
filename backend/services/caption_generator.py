"""
ClipAI — Auto Captions Service
Generates styled, synchronized ASS (Advanced SubStation Alpha) subtitles
for 9:16 vertical video clips.

Supported Languages:
- English, Hindi (Devanagari), and Punjabi (Gurmukhi).
- Uses Windows native 'Nirmala UI' font for flawless multi-script rendering.

Optimization:
- Reuses existing speech_segments from faster-whisper highlight detection
  wherever available to avoid redundant transcription and save CPU/RAM.
- Slices and formats speech into short, high-engagement 3–5 word subtitle cards.
"""

import re
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

from services.highlight_detector import SpeechSegment, get_whisper_model

logger = logging.getLogger(__name__)

# Native Windows font designed for all Indic and Latin scripts
ASS_FONT_NAME = "Nirmala UI"
FONT_SIZE = 58
OUTLINE_SIZE = 4
SHADOW_DEPTH = 2
MARGIN_VERTICAL = 240  # Sits comfortably above bottom mobile UI controls


def format_ass_time(seconds: float) -> str:
    """
    Formats a duration in seconds into ASS timestamp format: H:MM:SS.cs
    e.g. 75.345 -> '0:01:15.35'
    """
    total_sec = max(0.0, float(seconds))
    hrs = int(total_sec // 3600)
    mins = int((total_sec % 3600) // 60)
    secs = int(total_sec % 60)
    centis = int(round((total_sec - int(total_sec)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{centis:02d}"


def _split_into_snappy_cards(
    text: str,
    seg_start: float,
    seg_end: float,
    max_words_per_card: int = 5,
) -> List[Tuple[float, float, str]]:
    """
    Splits a longer speech segment into smaller, snappy subtitle cards
    (3 to 5 words) distributed proportionally across the segment's duration.
    Returns a list of (card_start, card_end, card_text).
    """
    words = text.strip().split()
    if not words:
        return []

    duration = max(0.4, seg_end - seg_start)

    # If already short, return as a single card
    if len(words) <= max_words_per_card:
        return [(round(seg_start, 2), round(seg_end, 2), " ".join(words))]

    # Group words into chunks of max_words_per_card
    chunks: List[List[str]] = []
    current_chunk: List[str] = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words_per_card:
            chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk)

    total_words = len(words)
    time_per_word = duration / total_words

    cards: List[Tuple[float, float, str]] = []
    current_time = seg_start

    for chunk in chunks:
        chunk_word_count = len(chunk)
        chunk_duration = chunk_word_count * time_per_word
        card_start = current_time
        card_end = min(seg_end, current_time + chunk_duration)
        cards.append((
            round(card_start, 2),
            round(card_end, 2),
            " ".join(chunk),
        ))
        current_time = card_end

    return cards


def build_ass_content(dialogue_cards: List[Tuple[float, float, str]]) -> str:
    """
    Constructs a complete ASS subtitle file string configured for a 1080x1920
    vertical canvas using Nirmala UI font.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{ASS_FONT_NAME},{FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{OUTLINE_SIZE},{SHADOW_DEPTH},2,40,40,{MARGIN_VERTICAL},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogue_lines: List[str] = []
    for start, end, text in dialogue_cards:
        # Sanitize ASS special formatting characters
        clean_text = text.replace("{", "(").replace("}", ")").replace("\\", "").strip()
        if not clean_text:
            continue
        start_str = format_ass_time(start)
        end_str = format_ass_time(end)
        dialogue_lines.append(
            f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{clean_text}"
        )

    return header + "\n".join(dialogue_lines) + "\n"


def generate_clip_captions_from_segments(
    speech_segments: List[SpeechSegment],
    clip_start: float,
    clip_duration: float,
    output_ass_path: Path,
) -> Optional[Path]:
    """
    Extracts speech within [clip_start, clip_start + clip_duration] from pre-existing
    Whisper speech segments, shifts timestamps relative to 0.0s, and writes an ASS file.
    Reuses transcription results without running Whisper again.
    """
    clip_end = clip_start + clip_duration
    all_cards: List[Tuple[float, float, str]] = []

    for seg in speech_segments:
        # Check overlap between segment and clip window
        overlap_start = max(clip_start, seg.start)
        overlap_end = min(clip_end, seg.end)

        if overlap_end > overlap_start:
            # Shift timestamps so they are relative to the start of this clip (0.0s)
            rel_start = max(0.0, overlap_start - clip_start)
            rel_end = min(clip_duration, overlap_end - clip_start)

            if rel_end > rel_start and seg.text.strip():
                cards = _split_into_snappy_cards(
                    text=seg.text,
                    seg_start=rel_start,
                    seg_end=rel_end,
                )
                all_cards.extend(cards)

    if not all_cards:
        logger.info("No speech dialogue found in clip window [%.1fs - %.1fs].", clip_start, clip_end)
        return None

    # Sort cards by start time
    all_cards.sort(key=lambda c: c[0])

    ass_text = build_ass_content(all_cards)
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    output_ass_path.write_text(ass_text, encoding="utf-8")

    logger.info(
        "Generated ASS subtitles with %d dialogue cards for clip (saved to %s).",
        len(all_cards), output_ass_path.name
    )
    return output_ass_path


def transcribe_and_generate_clip_captions(
    video_path: Path,
    clip_start: float,
    clip_duration: float,
    output_ass_path: Path,
) -> Optional[Path]:
    """
    Fallback method: Transcribes only the 25-second audio chunk of a clip on-demand
    if full video speech segments were not provided.
    Fast on CPU (~1-2 seconds) because audio is only 25 seconds long.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    temp_id = uuid.uuid4().hex[:8]
    temp_wav = output_ass_path.parent / f"_temp_clip_audio_{temp_id}.wav"

    try:
        # Extract audio for this 25s clip only
        cmd = [
            ffmpeg, "-y",
            "-ss", str(round(clip_start, 2)),
            "-t", str(clip_duration),
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(temp_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0 or not temp_wav.exists():
            return None

        model = get_whisper_model()
        segments_gen, _ = model.transcribe(
            str(temp_wav),
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        all_cards: List[Tuple[float, float, str]] = []
        for s in segments_gen:
            text = (s.text or "").strip()
            if not text:
                continue
            cards = _split_into_snappy_cards(
                text=text,
                seg_start=float(s.start),
                seg_end=float(s.end),
            )
            all_cards.extend(cards)

        if not all_cards:
            return None

        all_cards.sort(key=lambda c: c[0])
        ass_text = build_ass_content(all_cards)
        output_ass_path.write_text(ass_text, encoding="utf-8")
        return output_ass_path

    except Exception as exc:
        logger.warning("On-demand clip transcription failed: %s", exc)
        return None
    finally:
        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except Exception:
                pass
