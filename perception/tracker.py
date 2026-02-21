from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.detector import Detection, OpenCVPersonDetector, YOLODetector

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Track:
    person_id: int
    bbox: tuple[int, int, int, int]
    conf: float


def _center(b: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class CentroidTracker:
    def __init__(self, distance_threshold: float = 90.0) -> None:
        self._next_id = itertools.count(1)
        self._tracks: dict[int, tuple[int, int, int, int]] = {}
        self.distance_threshold = distance_threshold

    def update(self, detections: list[Detection]) -> list[Track]:
        assigned: set[int] = set()
        output: list[Track] = []
        for det in detections:
            c = _center(det.bbox)
            best_id = None
            best_dist = float("inf")
            for tid, tb in self._tracks.items():
                if tid in assigned:
                    continue
                tc = _center(tb)
                dist = math.hypot(c[0] - tc[0], c[1] - tc[1])
                if dist < best_dist and dist <= self.distance_threshold:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = next(self._next_id)
            self._tracks[best_id] = det.bbox
            assigned.add(best_id)
            output.append(Track(person_id=best_id, bbox=det.bbox, conf=det.conf))
        return output


class MultiObjectTracker:
    def __init__(self, yolo_model: str, yolo_conf: float) -> None:
        self._detector = YOLODetector(model_name=yolo_model, conf=yolo_conf)
        self._cv_detector = OpenCVPersonDetector()
        self._fallback_tracker = CentroidTracker()
        self._use_bytetrack = self._detector.enabled

    def update(self, frame: np.ndarray) -> list[Track]:
        if self._use_bytetrack and self._detector.model is not None:
            try:
                results = self._detector.model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=0.3,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )
                if not results:
                    return []
                boxes = results[0].boxes
                if boxes is None:
                    return []
                if boxes.id is None:
                    # BYTETrack can return detections without assigned IDs early in stream.
                    # Fall back to centroid assignment so events/alerts still work.
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
                    detections: list[Detection] = []
                    for b, c in zip(xyxy, confs):
                        x1, y1, x2, y2 = [int(v) for v in b[:4]]
                        detections.append(Detection(bbox=(x1, y1, x2, y2), conf=float(c), cls=0))
                    return self._fallback_tracker.update(detections)
                ids = boxes.id.int().cpu().tolist() if hasattr(boxes.id, "cpu") else boxes.id
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
                tracks: list[Track] = []
                for tid, b, c in zip(ids, xyxy, confs):
                    x1, y1, x2, y2 = [int(v) for v in b[:4]]
                    tracks.append(Track(person_id=int(tid), bbox=(x1, y1, x2, y2), conf=float(c)))
                return tracks
            except Exception as exc:
                LOGGER.warning("BYTETrack path failed. fallback centroid tracker enabled. reason=%s", exc)
                self._use_bytetrack = False

        detections = self._detector.detect(frame)
        if not detections:
            detections = self._cv_detector.detect(frame)
        return self._fallback_tracker.update(detections)
