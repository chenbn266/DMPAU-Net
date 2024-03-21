import json
import os
from glob import glob
from subprocess import call
import time

import nibabel
import numpy as np
from joblib import Parallel, delayed
import SimpleITK as sitk
# 加载nii.gz
def load_nifty(directory, example_id, suffix):
    return nibabel.load(os.path.join(directory, example_id + "-" + suffix + ".nii.gz"))

# 加载多模态
def load_channels(d, example_id):
    return [load_nifty(d, example_id, suffix) for suffix in ["t2f", "t1n", "t1c", "t2w"]]

def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        k = nifty.get_fdata()
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)
def get_data_itk(nifty,dtype="int16"):
    if dtype == "int16":
        data = sitk.ReadImage(r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/BraTS-GLI-01790-000/BraTS-GLI-01790-000-t2w.nii.gz')

def prepare_nifty(d):
    # print("dddd",d)
    example_id = d.split("/")[-1]
    # print("example_id",example_id)
    flair, t1, t1ce, t2 = load_channels(d, example_id)
    affine, header = flair.affine, flair.header
    tk = get_data(t1ce)
    tk22=get_data(t2)
    vol = np.stack([get_data(flair), get_data(t1), get_data(t1ce), get_data(t2)], axis=-1)
    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
    nibabel.save(vol, os.path.join(d, example_id + "_all.nii.gz"))

    if os.path.exists(os.path.join(d, example_id + "-seg.nii.gz")):
        seg = load_nifty(d, example_id, "seg")
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        vol[vol == 3] = 3
        seg = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
        nibabel.save(seg, os.path.join(d, example_id + "_seg.nii.gz"))




def prepare_dirs(data, train):
    # print("data",data)
    img_path, lbl_path = os.path.join(data, "images"), os.path.join(data, "labels")

    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(data, "BraTS*"))
    # print(dirs)

    for d in dirs:
        # print("dddd",d)
        if "-" in d.split("/")[-1]:
            # print(d.split("/")[-1])
            files = glob(os.path.join(d, "*.nii.gz"))
            # print(len(files))
            for f in files:
                # print(f)
                if "t2f" in f or "t1n" in f or "t1c" in f or "t2w" in f:
                    # print("fffffff")
                    continue
                if "_seg" in f:
                    call(f"mv {f} {lbl_path}", shell=True)
                if "_all" in f:
                    call(f"mv {f} {img_path}", shell=True)
        call(f"rm -rf {d}", shell=True)

def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images", "*")), glob(os.path.join(data, "labels", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    modality = {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
    labels_dict = {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }
    # data = data+"/dataset.json"
    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

# 并行计算加速
def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

def prepare_dataset(data, train):
    print(f"Preparing BraTS20 dataset from: {data}")
    start = time.time()
    # print(os.path.join(data  ,"BraTS*"))
    run_parallel(prepare_nifty, sorted(glob(os.path.join(data,"BraTS*"))))

    prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")


# prepare_dataset("/home/user/4TB/datasets/MICCAI_BraTS2020_TrainingData", True)
# prepare_dataset("/home/user/home/administrator/3Dbainian/MedicSegmentation/MICCAI_BraTS2020_TrainData", True)
# prepare_dataset("/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", True)
prepare_dataset("/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData", False)
# prepare_dataset("/home/user/4TB/datasets/MICCAI_BraTS2020_ValidationData", False)
print("Finished!")
