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
        infer_width: int = 960,
        infer_height: int = 540,
        imgsz: int = 960,
        grayscale: bool = False,
        class_names: Optional[List[str]] = None,
    ):
        self.config = config
        self.max_det = max_det
        self.verbose = verbose
        self.conf = conf
        # Pre-resize preserving 16:9, then let the model letterbox to a square.
        # These must correspond to the resolution the detector was TRAINED at:
        # feeding a 960-trained model at 640 shrinks the self-indicator below the
        # size the P2 head was trained to find, and nothing reports the mismatch.
        self.infer_width = int(infer_width)
        self.infer_height = int(infer_height)
        self.imgsz = int(imgsz)
        # The detector is trained on colour. Converting to grayscale here was a
        # workaround for a grayscale-trained model and destroys accuracy otherwise.
        self.grayscale = bool(grayscale)
        # Class names come from the model's own metadata by default. Hardcoding
        # them lets the list silently drift out of order with data.yaml, which
        # mislabels every detection without raising anything.
        self.class_names = list(class_names) if class_names is not None else None
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
        model_names = getattr(self.yolo, "names", None)
        print(f"[YOLO] Loaded model: {model_path.as_posix()}")
        if self.class_names is not None:
            print(f"[YOLO] Class names overridden: {self.class_names}")
        elif model_names:
            print(f"[YOLO] Class names from model metadata: {model_names}")
        else:
            # Without names, detections carry numeric ids, `detect_schema` matches
            # nothing, and the pipeline yields an empty game state every frame while
            # reporting success. Refuse to start rather than lie for an entire run.
            raise RuntimeError(
                f"{model_path.as_posix()} exposes no class names.\n"
                "The engine almost certainly failed to deserialise -- check the lines above\n"
                "for a TensorRT version error. Without names every detection is a numeric id,\n"
                "the 3-class schema matches nothing, and the agent would train on an empty\n"
                "observation without any error being raised.\n\n"
                "Fix by rebuilding the engine against the installed TensorRT:\n"
                "  python -m tools.check_capture --rebuild-engine\n"
                "or point at the .pt directly, or pass class_names=['character','indicator_self','weapon']."
            )

    @staticmethod
    def _name_from(source, cls_id: int) -> Optional[str]:
        if isinstance(source, dict):
            value = source.get(cls_id)
            return None if value is None else str(value)
        if isinstance(source, (list, tuple)) and 0 <= cls_id < len(source):
            return str(source[cls_id])
        return None

    def _resolve_class_name(self, cls_id: int, names_from_result) -> str:
        """Model metadata wins; an explicit override wins over that."""
        if self.class_names is not None:
            name = self._name_from(self.class_names, cls_id)
            if name is not None:
                return name
        for source in (names_from_result, getattr(self.yolo, "names", None)):
            name = self._name_from(source, cls_id)
            if name is not None:
                return name
        return str(cls_id)

    def _results_to_detections(self, results) -> List[Dict]:
        detections = []
        if results:
            for res in results:
                names_from_result = getattr(res, "names", None)
                for box in res.boxes:
                    cls_id = int(box.cls)
                    class_name = self._resolve_class_name(cls_id, names_from_result)

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

            if self.grayscale:
                gray = self._cv2.cvtColor(model_input, self._cv2.COLOR_BGR2GRAY)
                model_input = self._cv2.cvtColor(gray, self._cv2.COLOR_GRAY2BGR)

        kwargs = dict(max_det=self.max_det, verbose=self.verbose, conf=self.conf, imgsz=self.imgsz)
        try:
            results = self.yolo(model_input, **kwargs)
        except MemoryError:
            # TensorRT engine metadata version mismatch — fall back to ONNX or PT.
            fallback = self._find_fallback(self._model_path)
            print(f"[YOLO] TensorRT MemoryError — falling back to {fallback}")
            self.yolo = YOLO(str(fallback), task="detect")
            self._model_path = fallback
            results = self.yolo(model_input, **kwargs)
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
