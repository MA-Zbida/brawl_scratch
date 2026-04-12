from feature_extractor.yolo.extract import Extract
import cv2

extract = Extract()
frame = cv2.imread("image copy  ea s ea s as ededdddds6.png")
res = extract.predict(frame=frame)
print(res)