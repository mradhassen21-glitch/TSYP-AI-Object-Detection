import argparse, json, time
import cv2
from ultralytics import YOLO

# ---------- Config ----------
DEFAULT_WEIGHTS = r"C:\Users\GIGABYTE\yoloe\runs\detect\train-3\weights\best.pt"
CONF = 0.15
ALERT_COVERAGE = 0.15   # fraction of frame covered by debris that triggers BLOCKAGE alert
MQTT_HOST = None        # e.g. "localhost" to publish alerts, None to disable
MQTT_TOPIC = "infra/debris/alerts"


# ---------- Description logic ----------
def position_of(cx, cy, w, h):
    horiz = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    vert = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    return f"{vert}-{horiz}"


def severity_of(area_ratio, in_path):
    if area_ratio > 0.08 or (in_path and area_ratio > 0.03):
        return "HIGH"
    if area_ratio > 0.02:
        return "MEDIUM"
    return "LOW"


def describe(det, w, h):
    x1, y1, x2, y2 = det["bbox"]
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pos = position_of(cx, cy, w, h)
    in_path = (w / 3 < cx < 2 * w / 3) and (cy > h / 3)
    sev = severity_of(area_ratio, in_path)
    det.update(area_ratio=round(area_ratio, 4), position=pos, in_path=in_path, severity=sev)
    det["text"] = (f"{det['label']} ({det['conf']:.0%}) at {pos}, "
                   f"covers {area_ratio:.1%} of view, "
                   f"{'IN the main path' if in_path else 'outside main path'}, severity {sev}")
    return det


# ---------- Edge decision ----------
def decide(dets):
    coverage = sum(d["area_ratio"] for d in dets)
    if any(d["severity"] == "HIGH" for d in dets) or coverage > ALERT_COVERAGE:
        return "ALERT_BLOCKAGE", coverage
    if dets:
        return "WARNING_DEBRIS", coverage
    return "CLEAR", coverage


COLORS = {"LOW": (0, 200, 0), "MEDIUM": (0, 200, 255), "HIGH": (0, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video file path or 0 for webcam")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to best.pt")
    ap.add_argument("--save", default="output.mp4")
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("ERROR: cannot open video source:", args.source)
        return
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    client = None
    if MQTT_HOST:
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        client.connect(MQTT_HOST, 1883)

    log = open("detections.jsonl", "a")
    last_alert, t_prev = 0, time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, conf=CONF, verbose=False)[0]
        names = res.names
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            d = {"label": names[int(b.cls[0])], "conf": float(b.conf[0]), "bbox": (x1, y1, x2, y2)}
            dets.append(describe(d, w, h))

        status, coverage = decide(dets)

        for d in dets:
            x1, y1, x2, y2 = map(int, d["bbox"])
            c = COLORS[d["severity"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{d['label']} {d['severity']}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        now = time.time()
        fps_now = 1 / max(now - t_prev, 1e-6)
        t_prev = now
        banner = ((0, 0, 255) if status == "ALERT_BLOCKAGE"
                  else (0, 165, 255) if status == "WARNING_DEBRIS" else (0, 150, 0))
        cv2.rectangle(frame, (0, 0), (w, 34), banner, -1)
        cv2.putText(frame, f"{status} | objects: {len(dets)} | coverage: {coverage:.1%} | {fps_now:.1f} FPS",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if dets:
            for d in dets:
                print(d["text"])
            log.write(json.dumps({"t": now, "status": status, "coverage": round(coverage, 4),
                                  "objects": [{k: v for k, v in d.items() if k != "bbox"} for d in dets]}) + "\n")
        if status == "ALERT_BLOCKAGE" and now - last_alert > 5:
            last_alert = now
            print(">>> ACTION: alert sent, maintenance crew notified")
            if client:
                client.publish(MQTT_TOPIC, json.dumps({"status": status, "coverage": coverage,
                                                       "objects": [d["label"] for d in dets]}))

        out.write(frame)
        cv2.imshow("Edge debris detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    log.close()
    cv2.destroyAllWindows()
    print("Done. Saved:", args.save)


if __name__ == "__main__":
    main()