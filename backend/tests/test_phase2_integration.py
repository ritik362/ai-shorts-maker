"""
Phase 2 Integration Test Suite: AI Highlight Detection Pipeline
"""

import sys
import time
import unittest
from pathlib import Path
from fastapi import HTTPException

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.clipper import _calculate_even_starts, validate_clip_request
from services.task_manager import task_manager
from main import (
    process_video_endpoint,
    get_process_status_endpoint,
    download_clip,
    ProcessRequest,
)

DOWNLOADS_DIR = backend_dir / "downloads"
PUNJABI_VIDEO = "8_Parche_Baani_Sandhu_Gur_Sidhu_Gurneet_Dosanjh_Punjabi_Song_8WWv5_k8D14.mp4"
CHANT_VIDEO = "Rasraj_Ji_Maharaj_-_रम_रम_रटत_रटत_-_Rama_Rama_Ratate_Ratate__osF3xqqNcl8.mp4"


class TestPhase2Integration(unittest.TestCase):

    def test_even_starts_calculation(self):
        """Verify _calculate_even_starts produces correct non-overlapping timestamps."""
        starts_1 = _calculate_even_starts(clip_count=1, total_duration=100.0, clip_duration=25)
        self.assertEqual(starts_1, [37.5])

        starts_3 = _calculate_even_starts(clip_count=3, total_duration=100.0, clip_duration=25)
        self.assertEqual(len(starts_3), 3)
        self.assertEqual(starts_3, [0.0, 37.5, 75.0])
        # Verify no overlap
        for i in range(len(starts_3) - 1):
            self.assertLessEqual(starts_3[i] + 25, starts_3[i + 1])

    def test_endpoint_detection_mode_validation(self):
        """Verify POST /process accepts 'ai' and 'even', and rejects invalid mode."""
        # Valid 'ai'
        req_ai = ProcessRequest(filename=PUNJABI_VIDEO, clip_count=2, clip_duration=25, detection_mode="ai")
        resp_ai = process_video_endpoint(req_ai)
        self.assertEqual(resp_ai.status, "queued")
        self.assertEqual(resp_ai.detection_mode, "ai")

        # Valid 'even'
        req_even = ProcessRequest(filename=PUNJABI_VIDEO, clip_count=2, clip_duration=25, detection_mode="even")
        resp_even = process_video_endpoint(req_even)
        self.assertEqual(resp_even.status, "queued")
        self.assertEqual(resp_even.detection_mode, "even")

        # Default is 'ai' when omitted
        req_default = ProcessRequest(filename=PUNJABI_VIDEO, clip_count=2, clip_duration=25)
        resp_default = process_video_endpoint(req_default)
        self.assertEqual(resp_default.detection_mode, "ai")

        # Invalid mode rejected with 422
        with self.assertRaises(HTTPException) as ctx:
            req_invalid = ProcessRequest.model_construct(
                filename=PUNJABI_VIDEO, clip_count=2, clip_duration=25, detection_mode="invalid_mode"
            )
            process_video_endpoint(req_invalid)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_task_manager_stages_and_tracking(self):
        """Verify task_manager records detection_mode and tracks stages."""
        task_id = task_manager.submit_task(
            filename=PUNJABI_VIDEO,
            clip_count=2,
            clip_duration=25,
            processor_fn=lambda **kw: None,
            detection_mode="ai",
        )
        task = task_manager.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task["detection_mode"], "ai")

    def test_clips_preview_and_download_endpoints(self):
        """Verify GET /clips/{filename} works for inline preview and download."""
        # Find any existing processed clip
        processed_dir = backend_dir / "processed"
        clips = list(processed_dir.glob("*.mp4"))
        self.assertGreater(len(clips), 0, "Expected existing processed clips")
        clip_name = clips[0].name

        # Preview (download=False)
        prev_resp = download_clip(clip_name, download=False)
        self.assertEqual(prev_resp.status_code, 200)
        self.assertEqual(prev_resp.media_type, "video/mp4")
        self.assertEqual(prev_resp.headers.get("accept-ranges"), "bytes")
        self.assertTrue(prev_resp.headers.get("content-disposition", "").startswith("inline;"))

        # Download (download=True)
        dl_resp = download_clip(clip_name, download=True)
        self.assertEqual(dl_resp.status_code, 200)
        self.assertTrue(dl_resp.headers.get("content-disposition", "").startswith("attachment;"))


if __name__ == "__main__":
    unittest.main()
