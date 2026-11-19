import random, shutil
from pathlib import Path
from collections import defaultdict

random.seed(42)
M = Path("merged")
TARGET = {"valid": 75, "test": 35}   # desired rock images per split

def group_key(p):
    # Roboflow augmented copies share the same name before ".rf."
    # keep them together so near-duplicates don't leak between splits
    return p.name.split(".rf.")[0]

groups = defaultdict(list)
for img in (M / "train" / "images").glob("rock_*"):
    groups[group_key(img)].append(img)

keys = list(groups)
random.shuffle(keys)

def move(img, split):
    lbl = M / "train" / "labels" / (img.stem + ".txt")
    shutil.move(str(img), M / split / "images" / img.name)
    if lbl.exists():
        shutil.move(str(lbl), M / split / "labels" / lbl.name)

for split in ["valid", "test"]:
    have = len(list((M / split / "images").glob("rock_*")))
    moved = 0
    while keys and have + moved < TARGET[split]:
        for img in groups[keys.pop()]:
            move(img, split)
            moved += 1

for s in ["train", "valid", "test"]:
    print(s, "plastic:", len(list((M/s/"images").glob("plastic_*"))),
             "rock:", len(list((M/s/"images").glob("rock_*"))))