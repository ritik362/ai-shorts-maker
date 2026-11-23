"""
ClipAI — AI Highlight Detection Engine (Phase 1)
Identifies engaging moments, dialogue hooks, questions, and high-interest
segments from video audio using faster-whisper and heuristic NLP scoring.

Hardware Target:
- Windows PC, 8 GB RAM, CPU-only (no GPU/CUDA required).
- Model: faster-whisper 'base' with INT8 quantization, cpu_threads=4.
- Memory: ~250–400 MB peak footprint (cached singleton).

Multi-language Support:
- Auto-detects and scores English, Hindi, Punjabi, and mixed language dialogue.
- Graceful fallback when no speech is present.
"""

import re
import uuid
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ── Global Model Singleton (Lazy Loaded & Reused) ────────────────────────────
_whisper_lock = threading.Lock()
_whisper_model: Optional[WhisperModel] = None

# Base model with INT8 quantization is optimal for 8 GB RAM CPU execution
MODEL_SIZE = "base"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4


def get_whisper_model() -> WhisperModel:
    """
    Returns a thread-safe cached instance of the faster-whisper model.
    Loads on-demand on first use and reuses across subsequent calls.
    Avoids loading multiple models to strictly preserve RAM on 8 GB systems.
    """
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                logger.info(
                    "Loading faster-whisper '%s' model (device=cpu, compute_type=%s, threads=%d)...",
                    MODEL_SIZE, COMPUTE_TYPE, CPU_THREADS,
                )
                _whisper_model = WhisperModel(
                    MODEL_SIZE,
                    device="cpu",
                    compute_type=COMPUTE_TYPE,
                    cpu_threads=CPU_THREADS,
                )
                logger.info("faster-whisper model loaded and cached successfully.")
    return _whisper_model


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str
    words_count: int


@dataclass
class HighlightCandidate:
    start: float
    end: float
    duration: float
    score: float
    transcript_snippet: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class HighlightDetectionResult:
    success: bool
    fallback_needed: bool
    candidates: List[HighlightCandidate] = field(default_factory=list)
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    total_speech_segments: int = 0
    total_speech_duration: float = 0.0
    reason: Optional[str] = None
    speech_segments: List[SpeechSegment] = field(default_factory=list)


# ── Heuristic Pattern Dictionaries (English, Hindi, Punjabi) ──────────────────

# Questions (English, Hindi, Punjabi in Latin and native script)
QUESTION_PATTERNS = [
    # English
    r"\b(why|how|what|who|when|where|which|whose|whom)\b",
    r"\b(can you|could you|do you|did you|is it|are you|would you|have you|will you)\b",
    # Hindi / Urdu (Latin & Devanagari)
    r"\b(kya|kyun|kyu|kaise|kahan|kab|kaun|kisne|kisko|kitna)\b",
    r"(क्या|क्यों|कैसे|कहाँ|कब|कौन|किसने|किसको|कितना)",
    # Punjabi (Latin & Gurmukhi)
    r"\b(ki|kyun|kiven|kithon|kithan|kadon|kaun|kinna)\b",
    r"(ਕੀ|ਕਿਉਂ|ਕਿਵੇਂ|ਕਿੱਥੇ|ਕਦੋਂ|ਕੌਣ|ਕਿੰਨਾ)",
]

# Hooks and high-curiosity phrases
HOOK_PATTERNS = [
    # English
    r"\b(the secret is|did you know|most people|the truth is|the real reason)\b",
    r"\b(never do|always remember|stop doing|listen to this|check this out)\b",
    r"\b(here is why|here's why|the mistake|the problem is|one thing|the key to)\b",
    r"\b(imagine if|what happens if|you won't believe|nobody talks about)\b",
    # Hindi / Urdu
    r"\b(dekho|suno|dhyaan se suno|sabse bada|sabse pehle|asliyat|sach ye hai)\b",
    r"\b(kabhi mat|yaad rakhna|ek baat|yeh galti|raaz|saccha|dhyan do)\b",
    r"(देखो|सुनो|ध्यान से सुनो|सबसे बड़ा|सबसे पहले|असलियत|सच यह है|याद रखना|एक बात|यह गलती|राज)",
    # Punjabi
    r"\b(vekho|suno|dhyan naal|gal suno|asli gal|ik gal|kade na|yaad rakhiyo)\b",
    r"(ਵੇਖੋ|ਸੁਣੋ|ਧਿਆਨ ਨਾਲ|ਗੱਲ ਸੁਣੋ|ਅਸਲੀ ਗੱਲ|ਇੱਕ ਗੱਲ|ਕਦੇ ਨਾ|ਯਾਦ ਰੱਖਿਓ)",
]

# Strong statements, superlatives, and emphasis
EMPHASIS_PATTERNS = [
    # English
    r"\b(absolutely|definitely|literally|impossible|incredible|insane|crazy|crucial)\b",
    r"\b(biggest|worst|best|greatest|shocking|dangerous|unbelievable|mind-blowing)\b",
    r"\b(billion|million|hundred percent|number one|game changer)\b",
    # Hindi
    r"\b(bilkul|zaroor|pakka|zabardast|khatarnak|hairan|asambhava|shandar)\b",
    r"(बिल्कुल|ज़रूर|पक्का|ज़बरदस्त|खतरनाक|हैरान|असंभव|शानदार)",
    # Punjabi
    r"\b(bilkul|zaroor|pakka|khatarnak|zabardast|shandar|vadiya)\b",
    r"(ਬਿਲਕੁਲ|ਜ਼ਰੂਰ|ਪੱਕਾ|ਖ਼ਤਰਨਾਕ|ਜ਼ਬਰਦਸਤ|ਸ਼ਾਨਦਾਰ|ਵਧੀਆ)",
]

# Compile regular expressions once for performance
_COMPILED_QUESTIONS = [re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS]
_COMPILED_HOOKS = [re.compile(p, re.IGNORECASE) for p in HOOK_PATTERNS]
_COMPILED_EMPHASIS = [re.compile(p, re.IGNORECASE) for p in EMPHASIS_PATTERNS]


# ── Audio Extraction ──────────────────────────────────────────────────────────

def extract_audio_16k(video_path: Path, output_wav_path: Path) -> bool:
    """
    Extracts audio from video_path as a 16kHz mono PCM WAV file using FFmpeg.
    This format is optimal for Whisper and keeps temporary file size under ~10 MB
    for typical 10-minute videos.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is not installed or not found on PATH.")

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vn",                       # strip video stream
        "-acodec", "pcm_s16le",      # 16-bit uncompressed PCM
        "-ar", "16000",              # 16 kHz sample rate required by Whisper
        "-ac", "1",                  # 1 channel (mono)
        str(output_wav_path),
    ]

    logger.info("Extracting 16kHz mono audio from %s...", video_path.name)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if proc.returncode != 0:
        logger.error("FFmpeg audio extraction failed: %s", proc.stderr.strip()[-300:])
        return False

    return True


# ── Speech Transcription ─────────────────────────────────────────────────────

def transcribe_audio(
    wav_path: Path,
    language: Optional[str] = None,
) -> Tuple[List[SpeechSegment], Optional[str], Optional[float]]:
    """
    Transcribes audio using faster-whisper.
    Returns (segments, detected_language, language_probability).
    """
    model = get_whisper_model()

    # Transcribe with VAD filter to ignore prolonged background silence
    segments_gen, info = model.transcribe(
        str(wav_path),
        beam_size=1,            # beam_size=1 is fast on CPU with minimal accuracy loss
        language=language,      # None = auto-detect (English, Hindi, Punjabi, etc.)
        vad_filter=True,        # filter out pure silence
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    detected_lang = info.language if info else None
    lang_prob = info.language_probability if info else None

    segments: List[SpeechSegment] = []
    for s in segments_gen:
        text = (s.text or "").strip()
        if not text:
            continue
        words = len(text.split())
        segments.append(SpeechSegment(
            start=round(float(s.start), 2),
            end=round(float(s.end), 2),
            text=text,
            words_count=words,
        ))

    logger.info(
        "Transcription complete: %d segments, detected_language='%s' (prob=%.2f)",
        len(segments), detected_lang, lang_prob or 0.0
    )
    return segments, detected_lang, lang_prob


# ── Heuristic Scoring Engine ──────────────────────────────────────────────────

def _score_text_features(text: str) -> Tuple[float, List[str]]:
    """
    Evaluates text against heuristic patterns and returns (feature_score, reasons).
    """
    score = 0.0
    reasons: List[str] = []

    # 1. Question checks
    has_question_mark = "?" in text or "؟" in text
    has_question_words = any(p.search(text) for p in _COMPILED_QUESTIONS)
    if has_question_mark or has_question_words:
        score += 3.0
        reasons.append("Contains question/curiosity cue")

    # 2. Hook and high-curiosity phrases
    found_hooks = [p.pattern for p in _COMPILED_HOOKS if p.search(text)]
    if found_hooks:
        score += 4.0
        reasons.append(f"Dialogue hook detected ({len(found_hooks)} match)")

    # 3. Strong statements / emphasis / exclamation
    has_exclamation = "!" in text
    found_emphasis = [p.pattern for p in _COMPILED_EMPHASIS if p.search(text)]
    if has_exclamation or found_emphasis:
        score += 2.0
        reasons.append("Emphatic statement/superlative")

    return score, reasons


def _evaluate_candidate_windows(
    segments: List[SpeechSegment],
    total_duration: float,
    clip_duration: int = 25,
    stride_sec: float = 2.0,
) -> List[HighlightCandidate]:
    """
    Evaluates sliding windows across the duration of the video.
    Calculates speech density, dialogue engagement, and heuristic score.
    """
    if not segments or total_duration < clip_duration:
        return []

    candidates: List[HighlightCandidate] = []
    max_start = max(0.0, total_duration - clip_duration)
    current_start = 0.0

    while current_start <= max_start:
        current_end = current_start + clip_duration

        # Find all speech segments that overlap with this window
        overlapping: List[SpeechSegment] = []
        speech_time_in_window = 0.0
        window_texts: List[str] = []

        for seg in segments:
            # Overlap check
            overlap_start = max(current_start, seg.start)
            overlap_end = min(current_end, seg.end)
            if overlap_end > overlap_start:
                overlapping.append(seg)
                speech_time_in_window += (overlap_end - overlap_start)
                window_texts.append(seg.text)

        # Calculate speech density (ratio of spoken time to window length)
        speech_density = min(1.0, speech_time_in_window / float(clip_duration))
        total_words = sum(s.words_count for s in overlapping)
        combined_text = " ".join(window_texts)

        # Baseline filter: A good highlight needs sufficient dialogue (at least 15 words and 25% speech density)
        if speech_density >= 0.25 and total_words >= 12:
            feature_score, reasons = _score_text_features(combined_text)

            # Speech density contribution (scaled up to 4 points)
            density_score = speech_density * 4.0
            total_score = feature_score + density_score

            # Dialogue cadence bonus: Reward active, continuous conversation
            if total_words >= 30:
                total_score += 1.5
                reasons.append("High dialogue density")

            # Create candidate snippet (truncated to 120 chars for display)
            snippet = combined_text[:120].strip()
            if len(combined_text) > 120:
                snippet += "..."

            candidates.append(HighlightCandidate(
                start=round(current_start, 2),
                end=round(current_end, 2),
                duration=float(clip_duration),
                score=round(total_score, 2),
                transcript_snippet=snippet,
                reasons=reasons,
            ))

        current_start += stride_sec

    return candidates


def _select_non_overlapping_candidates(
    candidates: List[HighlightCandidate],
    clip_count: int,
    clip_duration: int,
) -> List[HighlightCandidate]:
    """
    Greedily selects the top non-overlapping candidate highlight windows.
    Guarantees no two selected windows overlap.
    Returns candidates sorted chronologically.
    """
    # Sort candidates by score descending
    sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

    selected: List[HighlightCandidate] = []

    for cand in sorted_candidates:
        if len(selected) >= clip_count:
            break

        # Check for overlap against all already selected candidates
        has_overlap = False
        for s in selected:
            # Two intervals [c.start, c.end] and [s.start, s.end] overlap if max(c.start, s.start) < min(c.end, s.end)
            if max(cand.start, s.start) < min(cand.end, s.end):
                has_overlap = True
                break

        if not has_overlap:
            selected.append(cand)

    # Sort the chosen candidates chronologically for natural video flow
    selected.sort(key=lambda c: c.start)
    return selected


# ── Public API Entry Point ────────────────────────────────────────────────────

def detect_highlights(
    video_path: Path,
    total_duration: float,
    clip_count: int = 3,
    clip_duration: int = 25,
    language: Optional[str] = None,
) -> HighlightDetectionResult:
    """
    Main entry point for AI Highlight Detection.

    Args:
        video_path: Absolute path to the source video file in backend/downloads/.
        total_duration: Video duration in seconds (from ffprobe).
        clip_count: Number of highlight clips requested (1–30).
        clip_duration: Duration of each clip in seconds (20–30).
        language: Optional language hint (e.g. 'en', 'hi', 'pa'). Default None for auto-detect.

    Returns:
        HighlightDetectionResult with candidates and fallback indicators.
    """
    if not video_path.exists() or not video_path.is_file():
        return HighlightDetectionResult(
            success=False,
            fallback_needed=True,
            reason=f"Source video file not found: {video_path.name}",
        )

    if total_duration < clip_duration:
        return HighlightDetectionResult(
            success=False,
            fallback_needed=True,
            reason=f"Video duration ({total_duration:.1f}s) is shorter than clip duration ({clip_duration}s).",
        )

    # Temporary mono 16kHz WAV file path
    temp_wav_id = uuid.uuid4().hex[:10]
    temp_wav_path = video_path.parent / f"_temp_audio_{temp_wav_id}.wav"

    try:
        # Step 1: Extract 16kHz mono audio via FFmpeg
        extraction_ok = extract_audio_16k(video_path, temp_wav_path)
        if not extraction_ok or not temp_wav_path.exists():
            return HighlightDetectionResult(
                success=False,
                fallback_needed=True,
                reason="Audio extraction failed. Falling back to even spacing.",
            )

        # Step 2: Transcribe speech with faster-whisper (Base INT8)
        segments, detected_lang, lang_prob = transcribe_audio(temp_wav_path, language=language)

        total_speech_time = sum(s.end - s.start for s in segments)

        # Step 3: Check if speech was detected
        # If video is silent or has < 5s total speech, trigger graceful fallback
        if not segments or total_speech_time < 5.0:
            logger.info(
                "No significant speech detected in %s (total speech: %.1fs). Returning fallback result.",
                video_path.name, total_speech_time
            )
            return HighlightDetectionResult(
                success=False,
                fallback_needed=True,
                detected_language=detected_lang,
                language_probability=lang_prob,
                total_speech_segments=len(segments),
                total_speech_duration=round(total_speech_time, 2),
                reason="No significant speech detected. Video is suitable for even-spaced clipping.",
            )

        # Step 4: Evaluate sliding candidate windows with heuristic scoring
        raw_candidates = _evaluate_candidate_windows(
            segments=segments,
            total_duration=total_duration,
            clip_duration=clip_duration,
            stride_sec=2.5,
        )

        if not raw_candidates:
            logger.info("No candidate windows met dialogue criteria. Returning fallback.")
            return HighlightDetectionResult(
                success=False,
                fallback_needed=True,
                detected_language=detected_lang,
                language_probability=lang_prob,
                total_speech_segments=len(segments),
                total_speech_duration=round(total_speech_time, 2),
                reason="Speech was too sparse or short to form highlight windows.",
            )

        # Step 5: Select top non-overlapping candidates
        selected_candidates = _select_non_overlapping_candidates(
            candidates=raw_candidates,
            clip_count=clip_count,
            clip_duration=clip_duration,
        )

        # If we couldn't find enough non-overlapping windows (e.g. video only had 1 speech event)
        if len(selected_candidates) < clip_count:
            logger.info(
                "Found %d highlight candidates, but %d were requested.",
                len(selected_candidates), clip_count
            )

        logger.info(
            "AI Highlight Detection succeeded: %d highlight windows selected for '%s'.",
            len(selected_candidates), video_path.name
        )

        return HighlightDetectionResult(
            success=True,
            fallback_needed=False,
            candidates=selected_candidates,
            detected_language=detected_lang,
            language_probability=lang_prob,
            total_speech_segments=len(segments),
            total_speech_duration=round(total_speech_time, 2),
            speech_segments=segments,
        )

    except Exception as exc:
        logger.error("Highlight detection encountered an unexpected error: %s", exc, exc_info=True)
        return HighlightDetectionResult(
            success=False,
            fallback_needed=True,
            reason=f"Detection error: {str(exc)}",
        )

    finally:
        # Step 6: Guaranteed temporary WAV cleanup in finally block
        if temp_wav_path and temp_wav_path.exists():
            try:
                temp_wav_path.unlink()
                logger.debug("Cleaned up temporary audio: %s", temp_wav_path.name)
            except Exception as clean_err:
                logger.warning("Could not delete temporary WAV %s: %s", temp_wav_path, clean_err)
