"""
ClipAI — Task Manager Service (Step 3)
In-memory job queue and task tracking for background video processing.

CRITICAL ARCHITECTURAL NOTES:
1. IN-MEMORY STORAGE: All task states and job metadata are stored purely in memory.
   Task state and progress history are LOST whenever the FastAPI backend server restarts.
2. STRICTLY SEQUENTIAL: Uses a single background worker thread reading from a thread-safe
   queue.Queue. Only ONE FFmpeg job runs at a time. Any subsequent submitted jobs remain
   in "queued" status until the active job completes or fails.
3. THREAD SAFETY: All reads and writes to the internal task dictionary are synchronized
   using a threading.Lock.
"""

import queue
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class TaskManager:
    """
    Thread-safe in-memory task manager with a single-worker background queue.
    """
    def __init__(self):
        # In-memory task dictionary: task_id -> task_state_dict
        # NOTE: Content is not persisted across server restarts.
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Single-worker queue guarantees strictly one FFmpeg job runs at a time
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("TaskManager background worker thread initialized.")

    def submit_task(
        self,
        filename: str,
        clip_count: int,
        clip_duration: int,
        processor_fn: Callable[..., Any],
        detection_mode: str = "ai",
        auto_captions: bool = True,
    ) -> str:
        """
        Creates and queues a new background video processing task.
        Returns a unique task_id immediately.
        """
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "filename": filename,
                "status": "queued",
                "progress": 0,
                "current_clip": 0,
                "total_clips": clip_count,
                "completed_clips": 0,
                "clips": [],
                "error": None,
                "detection_mode": detection_mode,
                "auto_captions": auto_captions,
                "created_at": now,
                "updated_at": now,
            }

        # Put job on sequential queue
        self._queue.put((task_id, filename, clip_count, clip_duration, processor_fn, detection_mode, auto_captions))
        logger.info(
            "Job queued: task_id=%s, file=%s, clips=%d, mode=%s, captions=%s",
            task_id, filename, clip_count, detection_mode, auto_captions
        )
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Thread-safely retrieves the current snapshot of a task's state.
        Returns None if task_id is not found.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                # Return a shallow copy of task dict and clips list
                copied = dict(task)
                copied["clips"] = list(task["clips"])
                return copied
        return None

    def _update_task(self, task_id: str, **kwargs) -> None:
        """Internal helper to atomically update task fields."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(kwargs)
                self._tasks[task_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    def _worker_loop(self) -> None:
        """
        Daemon worker loop. Pulls one job at a time from queue.Queue.
        Guarantees strictly sequential FFmpeg processing.
        """
        while True:
            try:
                task_id, filename, clip_count, clip_duration, processor_fn, detection_mode, auto_captions = self._queue.get()
                logger.info(
                    "Starting processing for task_id=%s (mode=%s, captions=%s)",
                    task_id, detection_mode, auto_captions
                )
                self._execute_job(
                    task_id, filename, clip_count, clip_duration, processor_fn, detection_mode, auto_captions
                )
            except Exception as exc:
                logger.error("Unhandled worker exception for task %s: %s", task_id, exc, exc_info=True)
                self._update_task(task_id, status="failed", error=str(exc))
            finally:
                self._queue.task_done()

    def _execute_job(
        self,
        task_id: str,
        filename: str,
        clip_count: int,
        clip_duration: int,
        processor_fn: Callable[..., Any],
        detection_mode: str = "ai",
        auto_captions: bool = True,
    ) -> None:
        """Runs the video processor and updates live task progress."""
        initial_status = "analyzing_highlights" if detection_mode == "ai" else "processing"
        initial_clip = 0 if detection_mode == "ai" else 1
        self._update_task(task_id, status=initial_status, current_clip=initial_clip)

        def progress_callback(
            current_clip: int = 0,
            total_clips: int = clip_count,
            completed_clip_info: Optional[Dict[str, Any]] = None,
            is_failed: bool = False,
            err_msg: Optional[str] = None,
            stage: Optional[str] = None,
        ) -> None:
            with self._lock:
                if task_id not in self._tasks:
                    return
                task = self._tasks[task_id]

                if stage:
                    task["status"] = stage
                elif is_failed:
                    task["status"] = "failed"
                    task["error"] = err_msg
                elif current_clip > 0:
                    task["status"] = "processing"

                if completed_clip_info:
                    task["clips"].append(completed_clip_info)
                    task["completed_clips"] = len(task["clips"])

                task["current_clip"] = min(current_clip, total_clips)
                if total_clips > 0:
                    task["progress"] = int((task["completed_clips"] / total_clips) * 100)

                task["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Run video processor
        result = processor_fn(
            filename=filename,
            clip_count=clip_count,
            clip_duration=clip_duration,
            progress_callback=progress_callback,
            detection_mode=detection_mode,
            auto_captions=auto_captions,
        )

        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if not result.success:
                    task["status"] = "failed"
                    task["error"] = result.reason or "One or more clips failed to process."
                else:
                    task["status"] = "completed"
                    task["progress"] = 100
                    task["current_clip"] = clip_count
                task["updated_at"] = datetime.now(timezone.utc).isoformat()
                logger.info("Task %s completed with status=%s", task_id, task["status"])


# Global singleton instance
task_manager = TaskManager()
