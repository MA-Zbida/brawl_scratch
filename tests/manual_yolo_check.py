"""Manual detector spot-check — NOT part of the automated suite.

Named without a `test_` prefix on purpose: it needs a GPU, the TensorRT engine and a
real screen, so pytest must not collect it. Run it by hand:

    python tests/manual_yolo_check.py
"""

from feature_extractor.yolo.extract import Extract
import cv2

extract = Extract()
frame = cv2.imread("image copy.png")
res = extract.predict(frame=frame)
print(res)