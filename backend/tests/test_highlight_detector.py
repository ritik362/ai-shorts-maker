"""
Automated Test Suite for ClipAI AI Highlight Detection Service (Phase 1)
"""

import sys
import unittest
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.highlight_detector import (
    _score_text_features,
    _evaluate_candidate_windows,
    _select_non_overlapping_candidates,
    get_whisper_model,
    detect_highlights,
    SpeechSegment,
    HighlightCandidate,
)


class TestHighlightDetector(unittest.TestCase):

    def test_english_heuristics(self):
        text = "Did you know that the secret is how you manage your focus?"
        score, reasons = _score_text_features(text)
        self.assertGreaterEqual(score, 7.0)
        self.assertTrue(any("question" in r.lower() for r in reasons))
        self.assertTrue(any("hook" in r.lower() for r in reasons))

    def test_hindi_heuristics(self):
        text = "देखो सबसे बड़ा राज यह है कि आप क्या चाहते हैं?"
        score, reasons = _score_text_features(text)
        self.assertGreaterEqual(score, 7.0)
        self.assertTrue(any("question" in r.lower() for r in reasons))
        self.assertTrue(any("hook" in r.lower() for r in reasons))

    def test_punjabi_heuristics(self):
        text = "ਵੇਖੋ ਗੱਲ ਸੁਣੋ ਇਹ ਕਿਉਂ ਹੋ ਰਿਹਾ ਹੈ?"
        score, reasons = _score_text_features(text)
        self.assertGreaterEqual(score, 7.0)
        self.assertTrue(any("question" in r.lower() for r in reasons))
        self.assertTrue(any("hook" in r.lower() for r in reasons))

    def test_emphasis_heuristics(self):
        text = "This is literally the best and most incredible achievement!"
        score, reasons = _score_text_features(text)
        self.assertGreaterEqual(score, 2.0)
        self.assertTrue(any("emphatic" in r.lower() for r in reasons))

    def test_non_overlapping_selection(self):
        dummy_segments = [
            SpeechSegment(start=5.0, end=15.0, text="Did you know what happens here? This is the secret.", words_count=10),
            SpeechSegment(start=16.0, end=28.0, text="Never do this mistake because it is dangerous!", words_count=12),
            SpeechSegment(start=40.0, end=55.0, text="The truth is simple. Everyone should listen carefully.", words_count=14),
            SpeechSegment(start=70.0, end=95.0, text="Why do people fail? Here is the real reason and lesson.", words_count=18),
        ]

        candidates = _evaluate_candidate_windows(
            segments=dummy_segments,
            total_duration=120.0,
            clip_duration=25,
            stride_sec=2.0,
        )
        self.assertGreater(len(candidates), 0)

        selected = _select_non_overlapping_candidates(candidates, clip_count=3, clip_duration=25)
        self.assertLessEqual(len(selected), 3)

        # Check strict non-overlap invariant
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                c1, c2 = selected[i], selected[j]
                overlap = max(c1.start, c2.start) < min(c1.end, c2.end)
                self.assertFalse(overlap, f"Overlap detected between [{c1.start}, {c1.end}] and [{c2.start}, {c2.end}]")

    def test_model_singleton_reuse(self):
        m1 = get_whisper_model()
        m2 = get_whisper_model()
        self.assertIs(m1, m2, "Model singleton was not reused across invocations")

    def test_non_existent_file_fallback(self):
        fake_path = backend_dir / "downloads" / "non_existent_fake_video.mp4"
        res = detect_highlights(
            video_path=fake_path,
            total_duration=100.0,
            clip_count=3,
            clip_duration=25,
        )
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_needed)

    def test_short_duration_fallback(self):
        fake_path = backend_dir / "downloads" / ".gitkeep"
        res = detect_highlights(
            video_path=fake_path,
            total_duration=10.0,  # less than clip_duration=25
            clip_count=1,
            clip_duration=25,
        )
        self.assertFalse(res.success)
        self.assertTrue(res.fallback_needed)


if __name__ == "__main__":
    unittest.main()
