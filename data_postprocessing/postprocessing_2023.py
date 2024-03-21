import os
from glob import glob
from subprocess import call

import nibabel as nib
import numpy as np
from scipy.ndimage import label


def to_lbl(pred, p,v):
    enh = pred[2]
    c1, c2, c3 = pred[0] > p[0], pred[1] > p[1], pred[2] > p[2]
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 3

    components, n = label(pred == 3)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 3
    if 0 < et.sum() and et.sum() < v and np.mean(enh[et]) < 0.90:  # 500 150 73 35 15  0.9
        pred[et] = 1

    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred

def prepare_preditions(e,p,v):
    fname = e[0].split("/")[-1].split(".")[0]
    preds = [np.load(f) for f in e]

    p = to_lbl(np.mean(preds, 0),p,v)
    # p =
    img = nib.load(
        f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2023_val/images/{fname}.nii.gz")  # BraTS2020_00001.nii.gz
    # BraTS20_Validation_001.nii.gz
    fname = fname.split("_")[-2]
    fname = fname.split("-")
    # print(fname[2:],fname)

    nib.save(
        nib.Nifti1Image(p, img.affine, header=img.header),
        os.path.join(save_path, "BraTS-GLI-" + fname[-2] +"-"+ fname[-1]+".nii.gz"),)
ps = [ [0.7, 0.3, 0.4]]
vs = [500]
version = 0
for p in ps:
    for v in vs:
        save_path = f"/home/user/4TB/Chenbonian/medic-segmention/BraTS2023_results/last-v{version}-1000-{v}-{p[0]}-{p[1]}-{p[2]}-tta"
        os.makedirs(save_path)
        preds = sorted(glob(f"/home/user/4TB/Chenbonian/medic-segmention/BraTS2023_results/predictions_last-v{version}_task=19_fold=0_tta"))
        examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
        print(f"Preparing final predictions v{version}-1000-{v}-{p[0]}-{p[1]}-{p[2]}-tta",len(examples))

        for e in examples:
            prepare_preditions(e,p,v)
        print("Finished!")
