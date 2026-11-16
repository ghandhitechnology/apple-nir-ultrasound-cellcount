# Applies the trained v13b classifier to a run the same way v13b treated the original 60 photos. The
# 48 embedding components come from the original PCA basis, either refitted with the full reference
# embeddings or loaded from weights/pca_v13b.joblib when those are absent. Run it with CC_FULL set
# to the photo folder: python apply_fixed_pca.py <run> <model> <out_dir> --embed --all-in-frame.

import os
import sys
from pathlib import Path

import joblib
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

assert "--embed" in sys.argv and "--all-in-frame" in sys.argv
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v13, feats2

REF = Path(os.environ.get("V13B_REF_EMBED", r"D:\cellcount_v11_full\rDs_0.5_3\embed.npz"))
SAVED_PCA = Path(__file__).resolve().parents[2] / "weights" / "pca_v13b.joblib"
DEFAULT_MODE = "combined" if REF.exists() else "saved"
MODE = sys.argv[sys.argv.index("--pca") + 1] if "--pca" in sys.argv else DEFAULT_MODE


def combined_matrix(z):
    zr = np.load(REF)
    new_imgs = set(np.unique(z["image"]))
    all_imgs = sorted(set(np.unique(zr["image"])) | new_imgs)
    I, ids, E = [], [], []
    for img in all_imgs:
        src = z if img in new_imgs else zr
        m = src["image"] == img
        I.append(src["image"][m])
        ids.append(src["id"][m])
        E.append(src["E"][m])
    return np.concatenate(I), np.concatenate(ids), np.concatenate(E)


def load_feats(run):
    F = feats2.all_features2(run)
    z = np.load(Path(run) / "embed.npz")
    if MODE == "saved":
        p = joblib.load(SAVED_PCA)
        img, ids, pcs = z["image"], z["id"], p.transform(z["E"].astype(np.float32))
    elif MODE == "transform":
        p = PCA(48, random_state=0).fit(np.load(REF)["E"].astype(np.float32))
        img, ids, pcs = z["image"], z["id"], p.transform(z["E"].astype(np.float32))
    else:
        img, ids, E = combined_matrix(z)
        pcs = PCA(48, random_state=0).fit_transform(E.astype(np.float32))
    Ed = pd.DataFrame(pcs, columns=[f"e{i}" for i in range(48)])
    Ed["image"] = img
    Ed["id"] = ids
    Ed = Ed[Ed.image.isin(set(F.image))]
    return F.merge(Ed, on=["image", "id"], how="left")


v13.load_feats = load_feats

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--") and x != MODE]
    print("pca mode", MODE)
    v13.apply(a[0], a[1], a[2])
