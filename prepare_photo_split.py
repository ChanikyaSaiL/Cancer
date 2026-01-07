import os, random, shutil

random.seed(42)

SRC_NRML = r"archive/Oral cancer Dataset 2.0/OC Dataset kaggle new/NON CANCER"
SRC_OSCC = r"archive/Oral cancer Dataset 2.0/OC Dataset kaggle new/CANCER"

DEST = r"photo_split"
splits = ["train", "val", "test"]
ratio = [0.7, 0.15, 0.15]

for s in splits:
    for c in ["Normal", "OSCC"]:
        os.makedirs(os.path.join(DEST, s, c), exist_ok=True)

def split_class(src, cls):
    files = [f for f in os.listdir(src) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    random.shuffle(files)
    n = len(files)
    t = int(ratio[0]*n)
    v = int(ratio[1]*n)

    mapping = {
        "train": files[:t],
        "val": files[t:t+v],
        "test": files[t+v:]
    }

    for split in splits:
        for f in mapping[split]:
            shutil.copy(os.path.join(src, f), os.path.join(DEST, split, cls, f))

split_class(SRC_NRML, "Normal")
split_class(SRC_OSCC, "OSCC")

print("Photo dataset safely split.")
