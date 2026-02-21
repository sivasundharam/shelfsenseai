from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    conf: float
    cls: int = 0


class OpenCVPersonDetector:
    """Fallback person detector using HOG descriptor."""

    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))
        detections: list[Detection] = []
        for (x, y, w, h), wgt in zip(boxes, weights):
            detections.append(Detection(bbox=(int(x), int(y), int(x + w), int(y + h)), conf=float(wgt), cls=0))
        return detections


class YOLODetector:
    def __init__(self, model_name: str, conf: float = 0.35) -> None:
        self._model: Any | None = None
        self._conf = conf
        self._enabled = False
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_name)
            self._enabled = True
            LOGGER.info("Loaded YOLO model: %s", model_name)
        except Exception as exc:
            LOGGER.warning("YOLO unavailable; detector fallback only. reason=%s", exc)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    @property
    def model(self) -> Any | None:
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self.enabled:
            return []
        res = self._model.predict(frame, classes=[0], conf=self._conf, verbose=False)
        out: list[Detection] = []
        if not res:
            return out
        boxes = res[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        for b, c in zip(xyxy, confs):
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            out.append(Detection(bbox=(x1, y1, x2, y2), conf=float(c), cls=0))
        return out
