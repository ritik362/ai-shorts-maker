"""
Unit Tests for ClipAI Caption Generator Service
"""

import sys
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.highlight_detector import SpeechSegment
from services.caption_generator import (
    format_ass_time,
    _split_into_snappy_cards,
    build_ass_content,
    generate_clip_captions_from_segments,
)


class TestCaptionGenerator(unittest.TestCase):

    def test_format_ass_time(self):
        """Verify ASS time formatting to H:MM:SS.cs format."""
        self.assertEqual(format_ass_time(0.0), "0:00:00.00")
        self.assertEqual(format_ass_time(1.5), "0:00:01.50")
        self.assertEqual(format_ass_time(65.25), "0:01:05.25")
        self.assertEqual(format_ass_time(3605.12), "1:00:05.12")

    def test_snappy_card_splitting(self):
        """Verify long dialogue is chunked into 3-5 word subtitle cards."""
        # Short phrase: stays 1 card
        cards_short = _split_into_snappy_cards("Hello world welcome", 0.0, 3.0)
        self.assertEqual(len(cards_short), 1)
        self.assertEqual(cards_short[0][2], "Hello world welcome")

        # Longer phrase: 12 words split into 3 cards of ~4-5 words
        long_text = "This is a much longer sentence that definitely needs to be split nicely"
        cards_long = _split_into_snappy_cards(long_text, 0.0, 6.0, max_words_per_card=5)
        self.assertEqual(len(cards_long), 3)
        self.assertEqual(cards_long[0][0], 0.0)
        self.assertAlmostEqual(cards_long[-1][1], 6.0, places=2)

    def test_english_hindi_punjabi_ass_rendering(self):
        """Verify ASS content with Nirmala UI font renders EN, HI, PA text cleanly."""
        dialogue = [
            (0.5, 2.5, "Did you know that focus is everything?"),
            (2.5, 5.0, "देखो सबसे बड़ा राज यह है कि आप क्या चाहते हैं।"),
            (5.0, 7.5, "ਵੇਖੋ ਗੱਲ ਸੁਣੋ ਇਹ ਕਿਉਂ ਹੋ ਰਿਹਾ ਹੈ।"),
        ]
        ass_str = build_ass_content(dialogue)

        self.assertIn("PlayResX: 1080", ass_str)
        self.assertIn("PlayResY: 1920", ass_str)
        self.assertIn("Nirmala UI", ass_str)
        self.assertIn("Did you know", ass_str)
        self.assertIn("देखो सबसे बड़ा राज", ass_str)
        self.assertIn("ਵੇਖੋ ਗੱਲ ਸੁਣੋ", ass_str)

    def test_speech_segment_offset_and_synchronization(self):
        """Verify segment timestamp shifting relative to clip start."""
        test_dir = backend_dir / "tests" / "temp_sub_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        out_ass = test_dir / "test_sync.ass"

        # Clip starts at 10.0s, lasts 25.0s
        segments = [
            SpeechSegment(start=5.0, end=9.0, text="Out of window before clip", words_count=5),
            SpeechSegment(start=12.0, end=17.0, text="First dialogue inside clip", words_count=4),
            SpeechSegment(start=20.0, end=26.0, text="Second dialogue inside clip", words_count=4),
            SpeechSegment(start=38.0, end=42.0, text="Out of window after clip", words_count=5),
        ]

        try:
            res_path = generate_clip_captions_from_segments(
                speech_segments=segments,
                clip_start=10.0,
                clip_duration=25.0,
                output_ass_path=out_ass,
            )
            self.assertIsNotNone(res_path)
            self.assertTrue(out_ass.exists())

            content = out_ass.read_text(encoding="utf-8")
            # Should NOT contain out-of-window text
            self.assertNotIn("Out of window", content)
            # Should contain inside dialogues
            self.assertIn("First dialogue inside clip", content)
            self.assertIn("Second dialogue inside clip", content)
            # Relative start: 12.0 - 10.0 = 2.00s -> '0:00:02.00'
            self.assertIn("0:00:02.00", content)
        finally:
            if out_ass.exists():
                out_ass.unlink()
            if test_dir.exists():
                try:
                    test_dir.rmdir()
                except Exception:
                    pass

    def test_non_speech_returns_none(self):
        """Verify clips with no speech return None so no blank captions are burned."""
        test_dir = backend_dir / "tests" / "temp_sub_test"
        out_ass = test_dir / "test_empty.ass"

        segments = [
            SpeechSegment(start=50.0, end=60.0, text="Far away speech", words_count=3),
        ]

        # Clip is [0s - 25s], speech is at [50s - 60s]
        res = generate_clip_captions_from_segments(
            speech_segments=segments,
            clip_start=0.0,
            clip_duration=25.0,
            output_ass_path=out_ass,
        )
        self.assertIsNone(res)
        self.assertFalse(out_ass.exists())


if __name__ == "__main__":
    unittest.main()
