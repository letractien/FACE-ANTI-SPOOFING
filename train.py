from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./yolov8n.pt')
    model.train(data='./Dataset/SplitData/dataOffline.yaml', epochs=300, imgsz=416, batch=16, device='0', patience=10)