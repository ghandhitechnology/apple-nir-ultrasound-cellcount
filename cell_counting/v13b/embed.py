# Computes ImageNet ResNet-18 embeddings for every object in a run, from a raw crop and an
# object-focused crop. It writes embed.npz with one 1024-value vector per object.

import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import torch, torchvision

FULL = Path(__import__("os").environ["CC_FULL"])
run = Path(sys.argv[1])
torch.set_num_threads(14)
m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = torch.nn.Identity()
m.eval()
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
R, S = 48, 112
imgs_out, ids_out, embs = [], [], []
for f in sorted((run / "labels").glob("*.npz")):
    img = f.stem
    z = np.load(f)
    lab = z["labels"]
    win = z["window"]
    im = cv2.imread(str(FULL / f"{img}.jpg"))[:, :, ::-1].astype(np.float32) / 255.0
    o = pd.read_csv(run / "objects" / f"{img}.csv")
    pad = np.pad(im, ((R, R), (R, R), (0, 0)), mode="reflect")
    labp = np.zeros((im.shape[0] + 2 * R, im.shape[1] + 2 * R), np.int32)
    labp[R + win[1] : R + win[3], R + win[0] : R + win[2]] = lab
    batch = []
    for _, q in o.iterrows():
        x, y = int(round(q.x)) + R, int(round(q.y)) + R
        c = pad[y - R : y + R, x - R : x + R]
        mk = (labp[y - R : y + R, x - R : x + R] == q.id).astype(np.uint8)
        mk = cv2.dilate(mk, np.ones((7, 7), np.uint8)).astype(np.float32)[..., None]
        foc = c * (0.35 + 0.65 * mk) + (1 - (0.35 + 0.65 * mk)) * c.mean((0, 1))
        for v in (c, foc):
            batch.append((cv2.resize(v, (S, S), interpolation=cv2.INTER_LINEAR) - mean) / std)
    X = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2).copy())
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            out.append(m(X[i : i + 512]).numpy())
    E = np.concatenate(out).reshape(len(o), 2 * 512)
    imgs_out += [img] * len(o)
    ids_out += o.id.tolist()
    embs.append(E.astype(np.float16))
    print(img, len(o), flush=True)
np.savez(run / "embed.npz", image=np.array(imgs_out), id=np.array(ids_out), E=np.concatenate(embs))
print("done")
