from ultralytics import YOLO  # type: ignore[attr-defined]
from typing import Optional, List, Dict
from pathlib import Path


class Extract:
    def __init__(
        self,
        yolo_model: str = "feature_extractor/yolo/best.engine",
        config=None,
        max_det: int = 5,
        verbose: bool = False,
        conf: float = 0.25,
        infer_width: int = 640,
        infer_height: int = 360,
        class_names: Optional[List[str]] = None,
    ):
        self.config = config
        self.max_det = max_det
        self.verbose = verbose
        self.conf = conf
        self.infer_width = int(infer_width)
        self.infer_height = int(infer_height)
        self.class_names = list(class_names) if class_names is not None else ["agent", "op", "op1", "op2", "weapons"]
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("opencv-python is required for TensorRT preprocessing") from exc
        self._cv2 = cv2

        model_path = Path(yolo_model)
        if not model_path.exists():
            raise RuntimeError(f"YOLO model file not found: {yolo_model}")

        self.yolo = YOLO(str(model_path), task="detect")
        self._model_path = model_path
        print(f"[YOLO] Loaded model: {model_path.as_posix()}")

    def _results_to_detections(self, results) -> List[Dict]:
        detections = []
        if results:
            for res in results:
                names_from_result = getattr(res, "names", None)
                for box in res.boxes:
                    cls_id = int(box.cls)
                    if 0 <= cls_id < len(self.class_names):
                        class_name = self.class_names[cls_id]
                    elif isinstance(names_from_result, dict):
                        class_name = str(names_from_result.get(cls_id, cls_id))
                    elif isinstance(names_from_result, list) and 0 <= cls_id < len(names_from_result):
                        class_name = str(names_from_result[cls_id])
                    else:
                        class_name = str(cls_id)

                    detections.append({
                        'class_name': class_name,
                        'bbox': box.xywhn[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf)
                    })
        return detections

    def predict(self, frame) -> List[Dict]:
        model_input = frame
        if frame is not None:
            if self.infer_width > 0 and self.infer_height > 0:
                model_input = self._cv2.resize(
                    frame,
                    (self.infer_width, self.infer_height),
                    interpolation=self._cv2.INTER_AREA,
                )

            gray = self._cv2.cvtColor(model_input, self._cv2.COLOR_BGR2GRAY)
            model_input = self._cv2.cvtColor(gray, self._cv2.COLOR_GRAY2BGR)

        try:
            results = self.yolo(model_input, max_det=self.max_det, verbose=self.verbose, conf=self.conf)
        except MemoryError:
            # TensorRT engine metadata version mismatch — fall back to ONNX or PT.
            fallback = self._find_fallback(self._model_path)
            print(f"[YOLO] TensorRT MemoryError — falling back to {fallback}")
            self.yolo = YOLO(str(fallback), task="detect")
            self._model_path = fallback
            results = self.yolo(model_input, max_det=self.max_det, verbose=self.verbose, conf=self.conf)
        return self._results_to_detections(results)

    @staticmethod
    def _find_fallback(engine_path: Path) -> Path:
        """Return the best non-.engine alternative in the same directory."""
        for ext in (".onnx", ".pt"):
            candidate = engine_path.with_suffix(ext)
            if candidate.exists():
                return candidate
        # Try any .onnx / .pt in the same dir
        parent = engine_path.parent
        for ext in (".onnx", ".pt"):
            matches = list(parent.glob(f"*{ext}"))
            if matches:
                return matches[0]
        raise RuntimeError(
            f"TensorRT engine failed and no .onnx/.pt fallback found in {parent}"
        )
