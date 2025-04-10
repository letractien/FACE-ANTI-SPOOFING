import os
import cv2
import cvzone
from time import time
from cvzone.FaceDetectionModule import FaceDetector

####################################
inputFolder = './Dataset/DataStandard'
outputFolderPath = './Dataset/DataCollect'
confidence = 0.50
save = True
blurThreshold = 0
offsetPercentageW = 10
offsetPercentageH = 20
floatingPoint = 6
####################################

if not os.path.exists(outputFolderPath):
    os.makedirs(outputFolderPath)

# Map class folders to labels
classMap = {
    'normal': 1,
    'spoof': 0
}

detector = FaceDetector()

# Duyệt qua từng class
for className, classID in classMap.items():
    classPath = os.path.join(inputFolder, className)
    imageFiles = [f for f in os.listdir(classPath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for imageName in imageFiles:
        imagePath = os.path.join(classPath, imageName)
        img = cv2.imread(imagePath)
        if img is None:
            print(f"Lỗi đọc ảnh: {imagePath}")
            continue

        imgOut = img.copy()
        img, bboxs = detector.findFaces(img, draw=False)

        listBlur = []
        listInfo = []

        if bboxs:
            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                score = bbox["score"][0]

                if score > confidence:
                    offsetW = (offsetPercentageW / 100) * w
                    offsetH = (offsetPercentageH / 100) * h

                    x = int(max(x - offsetW, 0))
                    y = int(max(y - offsetH * 3, 0))
                    w = int(w + offsetW * 2)
                    h = int(h + offsetH * 3.5)

                    imgFace = img[y:y + h, x:x + w]
                    if imgFace.size == 0:
                        continue

                    blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                    listBlur.append(blurValue > blurThreshold)

                    ih, iw, _ = img.shape
                    xc, yc = x + w / 2, y + h / 2
                    xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                    wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                    listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")
                else:
                    print(f"Điểm số không đủ: {imagePath}")

            # Save nếu ảnh đủ nét
            if save and all(listBlur):
                timeNow = str(time()).replace('.', '')
                saveImgPath = os.path.join(outputFolderPath, f"{timeNow}.jpg")
                saveTxtPath = os.path.join(outputFolderPath, f"{timeNow}.txt")

                cv2.imwrite(saveImgPath, imgOut)
                with open(saveTxtPath, 'w') as f:
                    f.writelines(listInfo)
            else:
                print(f"Ảnh không đủ nét: {imagePath}")

print("Xử lý hoàn tất.")
