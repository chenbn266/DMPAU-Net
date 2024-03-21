from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import matplotlib


# matplotlib.use('TkAgg')
# 需要查看的nii文件名文件名.nii或nii.gz
# filename = '/home/user/4TB/Chenbonian/medic-segmention/isnet/results/last-v5-1000-500-0.7-0.3-tta/master_00210.nii.gz'
# filename = '/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/KiTS2019_train/labels/segmentation_00151.nii.gz'
# filename = '/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/KiTS2019_val/images/case_00216.nii.gz'
filename = '/home/user/4TB/Chenbonian/medic-segmention/isnet/results/last-v27-1000-500-0.7-0.3-tta/prediction_00211.nii.gz'
img = nib.load(filename)
# pre = nib.load(pred)
# 打印文件信息
print(img)
print(img.dataobj.shape)
# print(pre.dataobj.shape)
# print(img.get_filename().shape())
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