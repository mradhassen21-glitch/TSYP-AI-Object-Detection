import argparse
import cv2
from ultralytics import YOLO

DEFAULT_WEIGHTS = r"C:\Users\GIGABYTE\yoloe\runs\detect\train-3\weights\best.pt"
CONF = 0.15

# one color per class (BGR)
COLORS = {"plastic": (0, 200, 0), "rock": (0, 140, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video file path or 0 for webcam")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--save", default="output.mp4")
    args = ap.parse_args()

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        print("ERROR: cannot open video source:", args.source)
        return
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, conf=CONF, verbose=False)[0]

        seen = set()
        for b in res.boxes:
            name = model.names[int(b.cls[0])]          # "plastic" or "rock"
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            color = COLORS.get(name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            seen.add(name)

        # console output: only the class names, nothing else
        for name in sorted(seen):
            print(name)

        out.write(frame)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()