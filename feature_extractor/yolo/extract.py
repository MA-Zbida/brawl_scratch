import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]
from typing import Optional, List, Dict, Any
from pathlib import Path


class Extract:
    def __init__(
        self,
        yolo_model: str = "feature_extractor/yolo/yolo26s_fp16.engine",
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
        if model_path.suffix.lower() != ".engine":
            raise RuntimeError(f"Only TensorRT engine models are supported. Got: {yolo_model}")
        if not model_path.exists():
            raise RuntimeError(f"TensorRT engine file not found: {yolo_model}")

        self.yolo = YOLO(str(model_path), task="detect")
        print(f"[YOLO] Loaded TensorRT engine: {model_path.as_posix()} (FP16)")

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

        results = self.yolo(model_input, max_det=self.max_det, verbose=self.verbose, conf=self.conf)
        return self._results_to_detections(results)

    def find_detection(self, detections, name) -> Optional[Dict[str, Any]]:
        candidates = [det for det in detections if det['class_name'] == name]
        if not candidates:
            return None
        
        return max(candidates, key=lambda d: d.get("confidence", 0.0))

    def find_detections(self, detections, name):
        return [det for det in detections if det['class_name'] == name]
    
    def detections_vector(self, detections) -> Optional[np.ndarray]:
        vec = np.zeros(shape=(7,), dtype=np.float32)

        # Find Detections
        agent = self.find_detection(detections=detections, name="agent")
        op = (self.find_detection(detections=detections, name="op") or
            self.find_detection(detections=detections, name="op1") or
            self.find_detection(detections=detections, name="op2"))
        weapons = self.find_detection(detections=detections, name="weapons")

        # Player 1 / Agent
        if agent:
            vec[0] = agent['bbox'][0]
            vec[1] = agent['bbox'][1]
        
        # Opponent
        if op:
            vec[2] = op['bbox'][0]
            vec[3] = op['bbox'][1]
            vec[4] = 0 if op['class_name'] == "op" else 1 if op['class_name'] == "op1" else 2
        
        # Weapons
        if weapons:
            vec[5] = weapons['bbox'][0]
            vec[6] = weapons['bbox'][1]

        return vec

