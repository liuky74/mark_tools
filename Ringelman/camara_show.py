import cv2
from Ringelman.blackness_detect import SubWindow
import numpy as np

window = SubWindow()

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

cap = cv2.VideoCapture("rtsp://admin:2Fenglan@192.168.0.231:554/h264/ch1/main/av_stream")
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    window.show(frame)
cv2.destroyAllWindows()
cap.release()