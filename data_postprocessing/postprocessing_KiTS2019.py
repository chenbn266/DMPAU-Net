import os
from glob import glob
from subprocess import call

import nibabel as nib
import numpy as np
from scipy.ndimage import label


def to_lbl(pred, p,v):
    print(pred.shape)
    # print(p[0])

    C,H,W,T = pred.shape
    pred = pred.argmax(0).astype(np.uint8)
    # print(pred.shape,"argmax")
    # print('1:', np.sum(pred == 1), ' | 2:', np.sum(pred == 2))
    seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

    seg_img[np.where(pred == 1)] = 1
    seg_img[np.where(pred== 2)] = 2
    # print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2))
    # seg_img = seg_img.astype(np.uint8)
    # print(seg_img.shape)


    # kidney,tumor = pred[0],pred[1]
    #
    # c1, c2 = kidney > p[0], tumor > p[1]
    # pred = (c1 > 0).astype(np.uint8)
    # pred = (c2 > 0).astype(np.uint8)
    # pred[(c2 == True) * (c1 == True)] = 2
    #
    #
    # components, n = label(pred == 4)
    # for et_idx in range(1, n + 1):
    #     _, counts = np.unique(pred[components == et_idx], return_counts=True)
    #     if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
    #         pred[components == et_idx] = 1
    #
    # et = pred == 4
    # if 0 < et.sum() and et.sum() < v and np.mean(enh[et]) < 0.90:  # 500 150 73 35 15  0.9
    #     pred[et] = 1
    #
    seg_img = np.transpose(seg_img, (2, 1, 0)).astype(np.uint8)
    print(seg_img.shape)
    return seg_img


def dataset_choise(fname,p,years):
    if years==2019:
        img = nib.load(
            f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/KiTS2019_val/images/{fname}.nii.gz")  # BraTS2020_00001.nii.gz
        # print(fname)
        fname = fname.split("_")
        print('ddd',fname)
        nib.save(
            nib.Nifti1Image(p, None, header=img.header),
            os.path.join(save_path, "prediction_" + fname[-1]+".nii.gz"),
        )
    if years==2023:
        img = nib.load(
            f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_val/images/{fname}.nii.gz")
        fname = fname.split("_")[-1]
        # print(fname[2:],fname)
        nib.save(
            nib.Nifti1Image(p, img.affine, header=img.header),
            os.path.join(save_path, "BraTS20_Validation_" + fname + ".nii.gz"),)
            # os.path.join(save_path, "BraTS20_Training_" + fname + ".nii.gz"),)
    if years == 2021:
        img = nib.load(f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2021_val/images/{fname}.nii.gz")
        fname = fname.split("_")[-1]
        nib.save(
            nib.Nifti1Image(p, img.affine, header=img.header),
            os.path.join(save_path, "BraTS21_" + fname + ".nii.gz"), )
def prepare_preditions(e,p,v,years,post):
    fname = e[0].split("/")[-1].split(".")[0]
    preds = [np.load(f) for f in e]
    print(len(preds))

    if post == 1:
        p = to_lbl(preds[0] ,p,v)


    dataset_choise(fname,p,years)
    # p =



post =1
if post == 1:
    # ps=[[0.5,0.5,0.5],[0.7,0.3,0.4]]
    ps = [[0.7, 0.3]]
    # ps = [[0.45,0.4,0.45]]
else : ps=[[0,0,0]]
# ps = [[0.5,0.5,0.5]]
vs = [500]
versions = [27]
years = 2019
fold = 0


for version in versions:
    for p in ps:
        for v in vs:
            save_path = f"/home/user/4TB/Chenbonian/medic-segmention/isnet/results/last-v{version}-1000-{v}-{p[0]}-{p[1]}-tta"
            # os.makedirs(save_path)


            if years==2019:
                preds = sorted(glob(f"/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_last-v{version}_task=23_fold={fold}_tta"))
                # preds = sorted(glob(
                # f"/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_epoch=379-dice=89_05_task=13_fold={fold}_tta"))


            examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
            print(f"Preparing final predictions v{version}-1000-{v}-{p[0]}-{p[1]}-tta",len(examples))

            for e in examples:

                prepare_preditions(e,p,v,years,post)
                print("Finished:",e)
            print("Finished!")
