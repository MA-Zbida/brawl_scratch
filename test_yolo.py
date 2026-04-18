from feature_extractor.yolo.extract import Extract
import cv2

extract = Extract()
frame = cv2.imread("image copy.png")
res = extract.predict(frame=frame)
print(res)