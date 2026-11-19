from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    model.train(data="C:/Users/GIGABYTE/Desktop/ai tsyp/merged/data.yaml",
                epochs=60, imgsz=640, batch=16, device=0, workers=2)