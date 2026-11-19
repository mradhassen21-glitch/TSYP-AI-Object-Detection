import yaml
from pathlib import Path

for folder in ["Stone Detection.v1i.yolov8",
               "Pebble Shell Dataset.v1-version-1.yolov8",
               "rock.v2-v6-detection-only.yolov8",
               "Plastic trash.v3i.yolov8"]:
    root = Path(folder)
    cfg = yaml.safe_load(open(root / "data.yaml"))
    print("\n", folder)
    print("  classes:", cfg["names"])
    for s in ["train", "valid", "test"]:
        d = root / s / "images"
        print("  ", s, len(list(d.iterdir())) if d.exists() else 0, "images")