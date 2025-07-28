import cv2
from ultralytics import YOLO
from mss import mss
import numpy as np
import pygetwindow as gw

model = YOLO('YOLOv11n_fire.pt')

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)

predict_args = {
    'imgsz': 160,
    'conf': 0.5,
    'device': 'cpu',
    'verbose': False
}

frame_counter = 0
skip_frames = 1

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_counter += 1
        if frame_counter % (skip_frames + 1) != 0:
            cv2.imshow('Fire Detection', frame)
            key = cv2.waitKey(1)
            if key == ord('q'): break
            continue
        
        #with mss() as sct:
            #win = gw.getWindowsWithTitle("fire-bbc.mp4")[0]
            #monitor = {"top": win.top, "left": win.left, "width": win.width, "height": win.height} 
            #screenshot = sct.grab(monitor)
            #frame = np.array(screenshot)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model.predict(frame, **predict_args)
        cv2.imshow('Fire Detection', results[0].plot())
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
