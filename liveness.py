import math
import time

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("./runs/detect/train/weights/best.pt")

classNames = ["fake", "real"]
prev_frame_time = 0
new_frame_time = 0
confidence = 0.6

def resize_img(img, width=416, height=416):
    imgin = img.copy()
    imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2RGB)
    M, N, C = imgin.shape

    if M > N:
        imgout = np.zeros((M, M, C), np.uint8) + 255
        imgout[:M, :N, :C] = imgin
        imgout = cv2.resize(imgout, (416, 416))
    elif M < N:
        imgout = np.zeros((N, N, C), np.uint8) + 255
        imgout[:M, :N, :C] = imgin
        imgout = cv2.resize(imgout, (416, 416))
    else:
        imgout = cv2.resize(imgin, (416, 416))

    return cv2.resize(img, (width, height))

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    img = resize_img(img, 416, 416)
    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])

            if conf > confidence:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,
                                   colorB=color)


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.imshow("Image", img)
    cv2.waitKey(1)