from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    runs = Path(r"C:\Users\GIGABYTE\yoloe\runs\detect")
    # newest last.pt among all runs
    last = max(runs.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    print("Resuming from:", last)
    YOLO(str(last)).train(resume=True)