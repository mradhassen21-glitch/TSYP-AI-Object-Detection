import shutil, yaml
from pathlib import Path

CLASSES = ["rock", "plastic"]
SPLITS = ["train", "valid", "test"]
OUT = Path("merged")

# folder -> (target class, names to KEEP, or None = keep all except EXCLUDE)
JOBS = [
    ("Plastic trash.v3i.yolov8", "plastic", None, {"glass bottle"}),
    ("rock.v2-v6-detection-only.yolov8", "rock", {"rock"}, set()),
    ("Stone Detection.v1i.yolov8", "rock", {"giok", "jasper"}, set()),
    ("Pebble Shell Dataset.v1-version-1.yolov8", "rock", {"pebble"}, set()),
]

for s in SPLITS:
    (OUT / s / "images").mkdir(parents=True, exist_ok=True)
    (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

for k, (folder, target, keep, exclude) in enumerate(JOBS):
    root = Path(folder)
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    ok_ids = {i for i, n in enumerate(names)
              if (n.lower() in keep if keep else n.lower() not in exclude)}
    new_id, tag = CLASSES.index(target), f"{target}_{k}_"
    print(folder, "-> keeping:", [names[i] for i in sorted(ok_ids)])

    for s in SPLITS:
        img_dir, lbl_dir = root / s / "images", root / s / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            lines = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5 or int(float(p[0])) not in ok_ids:
                        continue
                    if len(p) > 5:  # polygon -> bounding box
                        xs, ys = list(map(float, p[1::2])), list(map(float, p[2::2]))
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    else:
                        cx, cy, w, h = map(float, p[1:])
                    lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if not lines:      # nothing we want in this image -> skip it
                continue
            shutil.copy(img, OUT / s / "images" / (tag + img.name))
            (OUT / s / "labels" / (tag + img.stem + ".txt")).write_text("\n".join(lines))

cfg = {"path": str(OUT.resolve()), "train": "train/images", "val": "valid/images",
       "nc": len(CLASSES), "names": CLASSES}
if any((OUT / "test" / "images").iterdir()):
    cfg["test"] = "test/images"
yaml.safe_dump(cfg, open(OUT / "data.yaml", "w"))

print()
for s in SPLITS:
    imgs = list((OUT / s / "images").iterdir())
    print(s, "rock:", sum(i.name.startswith("rock_") for i in imgs),
             "plastic:", sum(i.name.startswith("plastic_") for i in imgs))