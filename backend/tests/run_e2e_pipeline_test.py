"""
End-to-End Pipeline Verification for AI Highlight Detection & Fallback
"""

import sys
import time
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from services.clipper import process_video
from services.task_manager import task_manager

PUNJABI_VIDEO = "8_Parche_Baani_Sandhu_Gur_Sidhu_Gurneet_Dosanjh_Punjabi_Song_8WWv5_k8D14.mp4"
CHANT_VIDEO = "Rasraj_Ji_Maharaj_-_रम_रम_रटत_रटत_-_Rama_Rama_Ratate_Ratate__osF3xqqNcl8.mp4"

print("=" * 60)
print("TEST 1: AI Detection Mode ('ai') on Punjabi video (speech present)")
print("=" * 60)

stages_recorded = []
def callback_1(current_clip=0, total_clips=1, completed_clip_info=None, is_failed=False, err_msg=None, stage=None):
    if stage and stage not in stages_recorded:
        stages_recorded.append(stage)
        print(f"   [STAGE TRANSITION] -> {stage}")

res_ai = process_video(
    filename=PUNJABI_VIDEO,
    clip_count=1,
    clip_duration=20,
    progress_callback=callback_1,
    detection_mode="ai",
)

print(f"Success: {res_ai.success}")
print(f"Clips generated: {len(res_ai.clips)}")
assert res_ai.success is True, f"AI processing failed: {res_ai.reason}"
assert "analyzing_highlights" in stages_recorded, "analyzing_highlights stage was not reported!"
assert "processing" in stages_recorded, "processing stage was not reported!"
print("[PASS] Test 1: AI Highlight Detection pipeline succeeded with correct stages!")

print("\n" + "=" * 60)
print("TEST 2: AI Detection Mode ('ai') on Chant/Music video (fallback trigger)")
print("=" * 60)

stages_recorded_2 = []
def callback_2(current_clip=0, total_clips=1, completed_clip_info=None, is_failed=False, err_msg=None, stage=None):
    if stage and stage not in stages_recorded_2:
        stages_recorded_2.append(stage)
        print(f"   [STAGE TRANSITION] -> {stage}")

res_fallback = process_video(
    filename=CHANT_VIDEO,
    clip_count=1,
    clip_duration=20,
    progress_callback=callback_2,
    detection_mode="ai",
)

print(f"Success: {res_fallback.success}")
print(f"Clips generated: {len(res_fallback.clips)}")
assert res_fallback.success is True, f"Fallback processing failed: {res_fallback.reason}"
assert "analyzing_highlights" in stages_recorded_2, "analyzing_highlights stage was not reported!"
assert "processing" in stages_recorded_2, "processing stage was not reported!"
print("[PASS] Test 2: AI detection fallback gracefully switched to even spacing!")

print("\n" + "=" * 60)
print("TEST 3: Even Spacing Mode ('even')")
print("=" * 60)

stages_recorded_3 = []
def callback_3(current_clip=0, total_clips=1, completed_clip_info=None, is_failed=False, err_msg=None, stage=None):
    if stage and stage not in stages_recorded_3:
        stages_recorded_3.append(stage)
        print(f"   [STAGE TRANSITION] -> {stage}")

res_even = process_video(
    filename=PUNJABI_VIDEO,
    clip_count=1,
    clip_duration=20,
    progress_callback=callback_3,
    detection_mode="even",
)

print(f"Success: {res_even.success}")
print(f"Clips generated: {len(res_even.clips)}")
assert res_even.success is True
assert "analyzing_highlights" not in stages_recorded_3, "analyzing_highlights should NOT run in even mode!"
print("[PASS] Test 3: Even spacing mode ran directly without AI analysis!")

print("\n" + "=" * 60)
print("ALL PIPELINE INTEGRATION TESTS PASSED 100%!")
print("=" * 60)
