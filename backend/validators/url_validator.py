"""
ClipAI — URL Validator
Validates YouTube and Instagram URLs without downloading or accessing content.
No DRM bypass, no login, no scraping.
"""

import re
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    valid: bool
    platform: Optional[str]
    url: str
    reason: Optional[str] = None


# ── YouTube Patterns ───────────────────────────────────────────────────────────
# Matches:
#   youtube.com/watch?v=VIDEO_ID
#   youtube.com/shorts/VIDEO_ID
#   youtu.be/VIDEO_ID
#   www.youtube.com/... (with or without www)
_YT_PATTERNS = [
    r"^(https?://)?(www\.)?youtube\.com/watch\?.*v=[\w-]{11}",
    r"^(https?://)?(www\.)?youtube\.com/shorts/[\w-]{11}",
    r"^(https?://)?youtu\.be/[\w-]{11}",
]

# ── Instagram Patterns ─────────────────────────────────────────────────────────
# Matches:
#   instagram.com/p/CODE/
#   instagram.com/reel/CODE/
#   instagram.com/tv/CODE/
#   www.instagram.com/... (with or without www)
_IG_PATTERNS = [
    r"^(https?://)?(www\.)?instagram\.com/p/[\w-]+/?",
    r"^(https?://)?(www\.)?instagram\.com/reel/[\w-]+/?",
    r"^(https?://)?(www\.)?instagram\.com/tv/[\w-]+/?",
]


def _matches_any(url: str, patterns: list[str]) -> bool:
    return any(re.match(p, url, re.IGNORECASE) for p in patterns)


def validate_url(raw_url: str) -> ValidationResult:
    """
    Validates a user-supplied URL and returns its platform.
    Does NOT make any network requests or access any content.
    """
    url = raw_url.strip()

    # ── Basic sanity check ────────────────────────────────────────────────────
    if not url:
        return ValidationResult(valid=False, platform=None, url=url,
                                reason="URL cannot be empty.")

    # Prepend scheme if missing so urlparse works correctly
    normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"

    try:
        parsed = urlparse(normalized)
        if not parsed.netloc:
            raise ValueError("No hostname")
    except Exception:
        return ValidationResult(valid=False, platform=None, url=url,
                                reason="That doesn't look like a valid URL.")

    # ── Platform detection ────────────────────────────────────────────────────
    if _matches_any(normalized, _YT_PATTERNS):
        return ValidationResult(valid=True, platform="youtube", url=normalized)

    if _matches_any(normalized, _IG_PATTERNS):
        return ValidationResult(valid=True, platform="instagram", url=normalized)

    # ── Recognized domain but wrong path ─────────────────────────────────────
    hostname = parsed.netloc.lower().replace("www.", "")

    if "youtube.com" in hostname or hostname == "youtu.be":
        return ValidationResult(
            valid=False, platform=None, url=url,
            reason="YouTube URL detected but it doesn't look like a valid video link. "
                   "Use a Watch, Shorts, or youtu.be link."
        )

    if "instagram.com" in hostname:
        return ValidationResult(
            valid=False, platform=None, url=url,
            reason="Instagram URL detected but it doesn't look like a valid post. "
                   "Use an /p/, /reel/, or /tv/ link."
        )

    # ── Completely unsupported domain ─────────────────────────────────────────
    return ValidationResult(
        valid=False, platform=None, url=url,
        reason=f"Unsupported platform. Please use a YouTube or Instagram link."
    )
