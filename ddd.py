import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob

n, z = 10, 97
data = sorted(glob("/home/user/4TB/Chenbonian/medic-segmention/final_4/*.nii.gz"))
data_u = sorted(glob("/home/user/4TB/Chenbonian/medic-segmention/u-4/*.nii.gz"))

for i in range(len(data)):
    fname = data[i].split("/")[-1].split(".")[0]
    print(fname)
    img = nib.load(f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/images/{fname}.nii.gz").get_fdata().astype(np.float32)
    ldd= nib.load(
        f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/labels/{fname}_seg.nii.gz").get_fdata().astype(np.uint8)[:, :, z]

    pred = nib.load(data[i]).get_fdata().astype(np.uint8)[:, :, z]
    pred_u = nib.load(data_u[i]).get_fdata().astype(np.uint8)[:, :, z]
    imgs = [img[:, :, z, i] for i in [0, 3]] +[ldd]+ [pred] +[pred_u]

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 12))
    for i in range(5):
        if i < 2:

            ax[i].imshow(imgs[i], cmap='gray')
        else:
            ax[i].imshow(imgs[i])
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()