import json
import os
from glob import glob
from subprocess import call
import time

import nibabel
import numpy as np
from joblib import Parallel, delayed

# 加载nii.gz
def load_nifty(directory, example_id, suffix):
    return nibabel.load(os.path.join(directory, example_id + "_" + suffix + ".nii.gz"))

# 加载多模态
def load_channels(d, example_id):
    return [load_nifty(d, example_id, suffix) for suffix in ["flair", "t1", "t1ce", "t2"]]

def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def prepare_nifty(d):
    # print("dddd",d)
    example_id = d.split("/")[-1]
    id = example_id.split("_")[-1]

    # print("example_id",example_id)
    # flair, t1, t1ce, t2 = load_channels(d, example_id)
    path =  "master_"+id+ ".nii.gz"
    image = nibabel.load(os.path.join(d,path))
    affine, header = image.affine, image.header
    vol = get_data(image)
    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
    nibabel.save(vol, os.path.join(d, example_id+ ".nii.gz"))

    if os.path.exists(os.path.join(d, example_id + "_seg.nii.gz")):
        seg = nibabel.load(os.path.join(d, "segmentation_"+id+".nii.gz"))
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        # vol[vol == 4] = 3
        seg = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
        nibabel.save(seg, os.path.join(d, example_id + "_seg.nii.gz"))




def prepare_dirs(data, train):
    print("data",data)
    img_path, lbl_path = os.path.join(data, "images"), os.path.join(data, "labels")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(data))

    for d in dirs:
        print("dddd",d)
        if "_" in d.split("/")[-1]:
            files = glob(os.path.join(d, "*.nii.gz"))
            for f in files:
                if "mast" in f:
                    call(f"mv {f} {img_path}", shell=True)

                if "seg" in f:
                    call(f"mv {f} {lbl_path}", shell=True)
                else:
                    continue
        # call(f"rm -rf {d}", shell=True)

def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images", "*")), glob(os.path.join(data, "labels", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    # modality = {"0": "", "1": "T1", "2": "T1CE", "3": "T2"}
    labels_dict = {"0": "background", "1": "kind", "2": "tumor"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        # "modality": modality,
        key: data_pairs,
    }
    # data = data+"/dataset.json"
    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

# 并行计算加速
def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

def prepare_dataset(data, train):
    print(f"Preparing Kits19 dataset from: {data}")
    start = time.time()
    # print(os.path.join(data  ,"BraTS*"))
    # run_parallel(prepare_nifty, sorted(glob(os.path.join(data ,"case*"))))

    # prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")



prepare_dataset("/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/KiTS2019_val/", False)
# prepare_dataset("/home/user/4TB/datasets/MICCAI_BraTS2020_ValidationData", False)
print("Finished!")
# data_path = "/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/kits19/data"
# da = sorted(glob(os.path.join(data_path ,"*")))
# print(da)
####移动文件
# def move_file(path,out_path):
#     for d in da:
#         if "master" in d:
#             id = d.split("/")[-1].split("_")[-1].split(".")[0]
#             ima = data_path+"/data"+"/case_"+id
#             call(f"mv {d} {ima}", shell=True)
# #####修改文件名
# def rename_file(path,out_path):
#     for d in da:
#         # print(d)
#         p = sorted(glob(os.path.join(d,"*")))
#         if len(p)>1:
#
#             id = p[1].split("/")[-2].split("_")[-1]
#             seg_name = data_path+"/case_"+id+"/segmentation_"+id +".nii.gz"
#             seg_old_name = data_path + "/case_" + id + "/segmentation.nii.gz"
#             call(f"mv {seg_old_name} {seg_name}", shell=True)
#             print(p)