"""
Unit tests for backend/services/downloader.py
Tests:
- Granular error classification (bot verification vs private vs login vs network vs unavailable vs rate-limited)
- Configuration verification (IPv4 binding, JS runtimes, player client fallbacks)
- URL extraction and safe filename generation
- Cache hit detection and completed video preservation
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.downloader import (
    classify_youtube_error,
    _get_base_ydl_opts,
    _get_js_runtimes,
    safe_filename,
    extract_video_id,
    download_video,
    DownloadResult,
)


class TestDownloaderClassification(unittest.TestCase):
    """Verifies that YouTube errors are accurately categorized without misclassifying as private videos."""

    def test_bot_verification_errors(self):
        err1 = Exception("ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you're not a bot. This helps protect our community.")
        err2 = Exception("ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you’re not a bot.")
        err3 = Exception("YouTube is requiring a captcha challenge before playback. errorScreen playerCaptchaViewModel")

        for err in [err1, err2, err3]:
            res = classify_youtube_error(err)
            self.assertIn("bot verification", res.lower())
            self.assertNotIn("private", res.lower())

    def test_private_video_errors(self):
        err1 = Exception("ERROR: [youtube] dQw4w9WgXcQ: Private video. Sign in if you've been granted access to this video")
        err2 = Exception("This video is private")

        for err in [err1, err2]:
            res = classify_youtube_error(err)
            self.assertIn("private", res.lower())
            self.assertNotIn("bot verification", res.lower())

    def test_login_and_age_gated_errors(self):
        err1 = Exception("ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm your age. This video may be inappropriate for some users.")
        err2 = Exception("This video is age-restricted and requires sign in.")
        err3 = Exception("Join this channel to get access to members-only perks")

        for err in [err1, err2, err3]:
            res = classify_youtube_error(err)
            self.assertIn("requires signing in", res.lower())
            self.assertNotIn("bot verification", res.lower())

    def test_unavailable_and_deleted_errors(self):
        err1 = Exception("ERROR: [youtube] dQw4w9WgXcQ: Video unavailable")
        err2 = Exception("This video is not available")
        err3 = Exception("This video has been removed by the uploader")

        for err in [err1, err2, err3]:
            res = classify_youtube_error(err)
            self.assertTrue(
                "unavailable" in res.lower() or "removed" in res.lower() or "deleted" in res.lower(),
                f"Unexpected classification for {err}: {res}"
            )
            self.assertNotIn("private", res.lower())

    def test_geo_restricted_errors(self):
        err = Exception("The uploader has not made this video available in your country")
        res = classify_youtube_error(err)
        self.assertIn("geographic", res.lower())
        self.assertNotIn("private", res.lower())

    def test_rate_limit_errors(self):
        err1 = Exception("ERROR: HTTP Error 429: Too Many Requests")
        err2 = Exception("This content isn't available, try again later. The current session has been rate-limited by YouTube")

        for err in [err1, err2]:
            res = classify_youtube_error(err)
            self.assertIn("rate limit", res.lower())
            self.assertNotIn("private", res.lower())

    def test_network_connection_errors(self):
        err1 = Exception("<urllib3.connection.HTTPSConnection>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed")
        err2 = Exception("Connection refused: [WinError 10060] The connection timed out")

        for err in [err1, err2]:
            res = classify_youtube_error(err)
            self.assertIn("network error", res.lower())
            self.assertNotIn("private", res.lower())

    def test_size_limit_errors(self):
        err = Exception("File is larger than max-filesize (550000000 > 524288000)")
        res = classify_youtube_error(err)
        self.assertIn("size limit", res.lower())


class TestDownloaderConfiguration(unittest.TestCase):
    """Verifies that yt-dlp is configured with IPv4, Node JS runtime, and player client fallbacks."""

    def test_get_js_runtimes(self):
        runtimes = _get_js_runtimes()
        self.assertIsInstance(runtimes, dict)
        # Node.js is installed on this PC
        self.assertIn("node", runtimes)

    def test_get_base_ydl_opts(self):
        opts = _get_base_ydl_opts()
        self.assertEqual(opts.get("source_address"), "0.0.0.0")  # Force IPv4
        self.assertIn("js_runtimes", opts)
        self.assertIn("node", opts["js_runtimes"])
        self.assertIn("extractor_args", opts)
        self.assertIn("youtube", opts["extractor_args"])
        self.assertIn("player_client", opts["extractor_args"]["youtube"])
        clients = opts["extractor_args"]["youtube"]["player_client"]
        self.assertTrue(len(clients) >= 4)
        self.assertIn("tv_samsung", clients)
        self.assertIn("web", clients)

    def test_safe_filename(self):
        fn = safe_filename("A Crazy Video! (1080p) / Best Moments", "dQw4w9WgXcQ")
        self.assertTrue(fn.endswith("_dQw4w9WgXcQ.mp4"))
        self.assertNotIn("/", fn)
        self.assertNotIn("!", fn)
        self.assertNotIn("(", fn)

    def test_extract_video_id(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")


class TestDownloaderCache(unittest.TestCase):
    """Verifies caching behavior."""

    @patch("services.downloader.find_completed_video")
    def test_cached_video_returns_without_network(self, mock_find):
        mock_file = MagicMock(spec=Path)
        mock_file.name = "Test_Video_dQw4w9WgXcQ.mp4"
        mock_file.stem = "Test_Video_dQw4w9WgXcQ"
        mock_stat = MagicMock()
        mock_stat.st_size = 1234567
        mock_file.stat.return_value = mock_stat
        mock_find.return_value = mock_file

        res = download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        self.assertTrue(res.success)
        self.assertEqual(res.filename, "Test_Video_dQw4w9WgXcQ.mp4")
        self.assertEqual(res.file_size, 1234567)


if __name__ == "__main__":
    unittest.main()
