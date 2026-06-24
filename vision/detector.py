"""YOLO object detector via ONNX Runtime.

Supports YOLOv8 / YOLOv11 ONNX models with mock fallback for
development without a GPU or model file.

Input:  BGR image (H, W, 3) as numpy array.
Output: list of ``Detection`` named tuples.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import numpy as np

Detection = namedtuple("Detection", ["class_name", "confidence", "box"])
"""A single detection result.

Fields
------
class_name : str
    Object class label (e.g. ``"person"``, ``"car"``).
confidence : float
    Detection confidence in [0, 1].
box : tuple[int, int, int, int]
    Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
"""

# COCO class names (80 classes) used by YOLOv8n
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class YOLODetector:
    """YOLO object detector via ONNX Runtime.

    Parameters
    ----------
    model_path : str or Path
        Path to YOLOv8/YOLOv11 ONNX model.
    classes : list[str]
        Class names (default: COCO 80 classes).
    confidence_threshold : float
        Minimum confidence to report a detection.
    iou_threshold : float
        IoU threshold for NMS (non-maximum suppression).
    input_size : tuple[int, int]
        Model input resolution ``(width, height)``.
    """

    def __init__(
        self,
        model_path: str | Path = "models/vision/yolov8n.onnx",
        classes: list[str] | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._model_path = Path(model_path)
        self._classes = classes or COCO_CLASSES
        self._conf_thresh = confidence_threshold
        self._iou_thresh = iou_threshold
        self._input_size = input_size
        self._model: object | None = None  # ort.InferenceSession
        self._is_loaded = False
        self._use_mock = False

    # -- Public API -------------------------------------------------------
    def load(self) -> None:
        """Load the ONNX model. Falls back to mock if model file is missing."""
        if self._is_loaded:
            return

        if not self._model_path.exists():
            self._use_mock = True
            self._is_loaded = True
            return

        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._gpu_available() else ["CPUExecutionProvider"]
        self._model = ort.InferenceSession(
            str(self._model_path),
            providers=providers,
        )
        self._is_loaded = True

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image of shape ``(H, W, 3)``.

        Returns
        -------
        list[Detection]
            Detected objects sorted by confidence (descending).
        """
        if self._use_mock or self._model is None:
            return self._mock_detect(image)

        # Preprocess
        input_tensor, scale, pad = self._preprocess(image)

        # Inference
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: input_tensor})

        # Postprocess
        detections = self._postprocess(outputs[0], image.shape[:2], scale, pad)
        return detections

    def unload(self) -> None:
        """Release resources."""
        self._model = None
        self._is_loaded = False

    # -- Properties -------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return self._classes

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_mock(self) -> bool:
        return self._use_mock

    # -- Preprocessing ----------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize + pad to model input size. Returns (tensor, scale, pad)."""
        import cv2

        h0, w0 = image.shape[:2]
        tw, th = self._input_size

        # Letterbox: resize keeping aspect ratio, pad to square
        scale = min(tw / w0, th / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_w = (tw - nw) // 2
        pad_h = (th - nh) // 2

        padded = np.full((th, tw, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        # BGR → RGB, HWC → CHW, normalize to [0, 1], add batch dim
        tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale, (pad_w, pad_h)

    # -- Postprocessing ---------------------------------------------------
    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        """Parse YOLO output into Detection list with NMS."""
        h0, w0 = original_shape
        tw, th = self._input_size
        pad_w, pad_h = pad

        # YOLOv8 output shape: [1, 84, 8400]
        # First 4 channels: bbox (cx, cy, w, h) normalized to input size
        # Remaining 80 channels: class scores
        if output.ndim == 3:
            output = output[0]  # squeeze batch dim

        output = output.T  # [8400, 84]

        boxes_xywh = output[:, :4]  # cx, cy, w, h (normalized)
        scores = output[:, 4:]      # class scores

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences > self._conf_thresh
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert cxcywh → xyxy in input coordinates
        boxes_xyxy = self._cxcywh_to_xyxy(boxes_xywh)

        # Scale from input coords → original image coords
        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h
        boxes_xyxy /= scale

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w0)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h0)

        # NMS
        keep = self._nms(boxes_xyxy, confidences, self._iou_thresh)
        boxes_xyxy = boxes_xyxy[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # Build results
        detections = []
        for box, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
            cls_id = int(cls_id)
            class_name = self._classes[cls_id] if cls_id < len(self._classes) else f"cls_{cls_id}"
            detections.append(Detection(
                class_name=class_name,
                confidence=float(conf),
                box=tuple(int(v) for v in box),
            ))

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    # -- Helpers ----------------------------------------------------------
    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
        out = np.zeros_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return out

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Simple NMS — keep highest-score boxes, suppress overlapping ones."""
        if len(boxes) == 0:
            return np.array([], dtype=int)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

            order = order[1:][iou < iou_threshold]

        return np.array(keep, dtype=int)

    @staticmethod
    def _gpu_available() -> bool:
        """Check if CUDAExecutionProvider is available."""
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    # -- Mock -------------------------------------------------------------
    def _mock_detect(self, image: np.ndarray) -> list[Detection]:
        """Mock detector — returns a fake detection ~10% of frames."""
        if np.random.random() > 0.1:
            return []
        cls_name = np.random.choice(["person", "car", "dog", "cat", "laptop"])
        h, w = image.shape[:2]
        return [Detection(
            class_name=cls_name,
            confidence=0.6 + np.random.random() * 0.35,
            box=(w // 4, h // 4, 3 * w // 4, 3 * h // 4),
        )]
