from __future__ import annotations

import time
from collections import deque
from typing import Callable


class PresenceGate:
    def __init__(self, window: int = 8, required_hits: int = 5, cooldown_seconds: int = 30):
        if required_hits > window:
            raise ValueError("required_hits cannot exceed window")
        self.samples: deque[bool] = deque(maxlen=window)
        self.required_hits = required_hits
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered = float("-inf")

    def observe(self, present: bool, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        self.samples.append(present)
        if len(self.samples) < self.samples.maxlen or sum(self.samples) < self.required_hits:
            return False
        if current - self.last_triggered < self.cooldown_seconds:
            return False
        self.last_triggered = current
        self.samples.clear()
        return True


class PresenceMonitor:
    """In-memory person-presence monitor. Frames are never written to disk."""

    def __init__(self, gate: PresenceGate, on_presence: Callable[[], None],
                 model_path: str = "yolov5s.pt", camera_index: int = 0):
        self.gate = gate
        self.on_presence = on_presence
        self.model_path = model_path
        self.camera_index = camera_index
        self.running = False

    def run(self) -> None:
        import cv2
        from ultralytics import YOLO

        model, camera = YOLO(self.model_path), cv2.VideoCapture(self.camera_index)
        self.running = True
        try:
            while self.running and camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    break
                result = model.predict(frame, classes=[0], verbose=False)[0]
                present = bool(result.boxes is not None and len(result.boxes) > 0)
                if self.gate.observe(present):
                    self.on_presence()
        finally:
            camera.release()

    def stop(self) -> None:
        self.running = False
