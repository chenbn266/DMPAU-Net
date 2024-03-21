import json
import os
from glob import glob
from subprocess import call
import time

import nibabel
import numpy as np
from joblib import Parallel, delayed

# 加载nii.gz
def load_nifty(directory, example_id):
    # print("loda")
    return nibabel.load(os.path.join(directory))

def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def prepare_nifty(d):
    print("dddd",d)
    example_id = d.split("/")[-1]
    print("example_id", example_id)
    if "vol" in example_id:
        print("example_id",example_id)
        flair = load_nifty(d, example_id)
        affine, header = flair.affine, flair.header
        vol = nibabel.nifti1.Nifti1Image(get_data(flair), affine, header=header)
        print("example_id", example_id)
        nibabel.save(vol, os.path.join("/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/LiTS2017/training/", example_id + ".gz"))
    # vol = np.stack([get_data(flair), get_data(t1), get_data(t1ce), get_data(t2)], axis=-1)
    # vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
        nibabel.save(vol, os.path.join(d))
    if "segmentation" in example_id:
        seg = load_nifty(d, example_id)
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")

        seg = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
        nibabel.save(seg, os.path.join("/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/LiTS2017/training/", example_id + ".gz"))

        # nibabel.save(seg, os.path.join(d))




def prepare_dirs(data, train):
    # print("data",data)
    img_path, lbl_path = os.path.join(data, "images"), os.path.join(data, "labels")
    call(f"mkdir {img_path}", shell=True)
    if train:
        call(f"mkdir {lbl_path}", shell=True)
    dirs = glob(os.path.join(data, "*"))

    for d in dirs:
        # print("dddd",d)
        if "_" in d.split("/")[-1]:
            files = glob(os.path.join(d, "*.nii"))
            for f in files:
                if "vol" in f:
                    continue
                if "seg" in f:
                    call(f"mv {f} {lbl_path}", shell=True)
                else:
                    call(f"mv {f} {img_path}", shell=True)
        # call(f"rm -rf {d}", shell=True)

def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images", "*")), glob(os.path.join(data, "labels", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    # modality = {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
    labels_dict = {"0": "background", "1": "live", "2": "tumor"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        key: data_pairs,
    }
    # data = data+"/dataset.json"
    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

# 并行计算加速
def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)

def prepare_dataset(data, train):
    print(f"Preparing LiTS dataset from: {data}")
    start = time.time()
    print(sorted(glob(os.path.join(data ,"*"))))
    run_parallel(prepare_nifty, sorted(glob(os.path.join(data ,"*"))))

    prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    print(f"Preparing time: {(end - start):.2f}")



# prepare_dataset("/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/LiTS2017/training", True)
# prepare_dataset("/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/LiTS2017/segmentetion", False)
print("Finished!")

seg = nibabel.load('/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/LiTS2017/training/segmentation-0.nii')
affine, header = seg.affine, seg.header
# vol = get_data(seg, "unit8")
# data = img.get_fdata()
print(seg.shape,affine,header)

from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import matplotlib


# matplotlib.use('TkAgg')
# 需要查看的nii文件名文件名.nii或nii.gz
# filename = '/home/zjx/3Dhetao/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_2013_2_1/BraTS19_2013_2_1_flair.nii.gz'
img = seg
# 打印文件信息
print(img)
print(img.dataobj.shape)
#shape不一定只有三个参数，打印出来看一下
width, height, queue = img.dataobj.shape
# 显示3D图像
OrthoSlicer3D(img.dataobj).show()
# 计算看需要多少个位置来放切片图
x = int((queue/10) ** 0.5) + 1
num = 1
# 按照10的步长，切片，显示2D图像
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(x, x, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
plt.show()