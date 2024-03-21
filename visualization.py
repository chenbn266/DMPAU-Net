#  encoding: utf-8
'''
@version: v1.0
@author: JiangZongKang
@time: 2019/10/7 17:07
@email: jiangzk2018@gmail.com
'''

import numpy as np
from skimage import color
from skimage.exposure import adjust_gamma
from skimage.util import img_as_float
import SimpleITK as sitk
import os
from skimage import io
import matplotlib.pyplot as plt
import imageio
from imgaug import augmenters as iaa
import numpy as np
from datetime import datetime

import nibabel as nib

def show_segmented_image(orig_img, pred_img):
    # Show the prediction over the original image
    # INPUT:
    #     1)orig_img: the test image, which was used as input
    #     2)pred_img: the prediction output
    # OUTPUT:
    #     segmented image rendering

    orig_img = sitk.GetArrayFromImage(sitk.ReadImage(orig_img))
    # orig_img = orig_img[0,:]
    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_img))
    orig_img = orig_img / np.max(orig_img)
    ones = np.argwhere(pred_img == 1)
    twos = np.argwhere(pred_img == 2)
    fours = np.argwhere(pred_img == 4)
    gray_img = img_as_float(orig_img)
    image = adjust_gamma(color.gray2rgb(gray_img), 1)
    sliced_image = image.copy()
    red_multiplier = [1, 0.2, 0.2]
    yellow_multiplier = [1, 1, 0.25]
    green_multiplier = [0.35, 0.75, 0.25]
    # pre=adjust_gamma(color.gray2rgb(pred_img), 1)
    # pred_img[pred_img==4]=[1, 1, 0.25]
    # pred_img[pred_img==2]=green_multiplier
    # pred_img[pred_img == 1] = red_multiplier

    # change colors of segmented classes
    for i in range(len(ones)):
        sliced_image[ones[i][0]][ones[i][1]][ones[i][2]] = red_multiplier
        # pre[ones[i][0]][ones[i][1]][ones[i][2]] = red_multiplier

    for i in range(len(twos)):
        sliced_image[twos[i][0]][twos[i][1]][twos[i][2]] = green_multiplier
        # pre[twos[i][0]][twos[i][1]][twos[i][2]] = green_multiplier
    for i in range(len(fours)):
        sliced_image[fours[i][0]][fours[i][1]][fours[i][2]] = yellow_multiplier
        # pre[fours[i][0]][fours[i][1]][fours[i][2]] = yellow_multiplier
    return sliced_image,pred_img


def save_visual_examples(input_dir,input_lab_dir ,pre_u_dir,pre_final_dir,output_dir, case_number, z_number, save):
    save_dir = os.path.join(output_dir, 'case_images/case_{}'.format(case_number))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("ddd")
    # for i in ['flair', 't1', 't1ce', 't2']:
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(input_dir + '_{}.nii.gz'.format(i)))
    #     img = img / np.max(img)
    #     gray_img = img_as_float(img)
    #     img = adjust_gamma(color.gray2rgb(gray_img), 1)
    #     z = z_number
    #     slice = img[z, :, :]
    #     io.imsave(save_dir + '/{}.png'.format(i), slice)
    #
    # gt专用

    img= sitk.GetArrayFromImage(sitk.ReadImage(input_dir+'.nii.gz'))
    print(img.shape)
    for i in range(4):
        input = img[i]
        print(input.shape)
        input = input/np.max(input)
        gray_img = img_as_float(input)
        gray_img = adjust_gamma(color.gray2rgb(gray_img),1)
        # slice = gray_img[z_number,:,:]

        slice = gray_img[ :, :,z_number]
        # slice = gray_img[:,  z_number,:]
        io.imsave(save_dir+'/L_input{}.png'.format(i),slice)

    # gt ,gt_s= show_segmented_image("/home/user/4TB/datasets/MICCAI_BraTS2020_TrainingData/"+case_number+"/"+case_number+'_flair.nii.gz',"/home/user/4TB/datasets/MICCAI_BraTS2020_TrainingData/"+case_number+"/"+case_number+ '_seg.nii.gz')
    gt, gt_s = show_segmented_image(
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_flair.nii.gz',
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_seg.nii.gz')

    # gt = sitk.GetArrayFromImage(sitk.ReadImage(input_lab_dir + '_seg.nii.gz'))
    print(gt.shape)
    # gt = gt[z_number, :, :]
    gt = gt[:, :,z_number]
    # gt = gt[:,z_number,  :]


    # gt = (gt * 255.0).astype('uint8')
    # gt_s = gt_s[z_number, :, :]
    # gt_s = (gt_s * 255.0).astype('uint8')
    io.imsave(save_dir + '/L_gt.png', gt)

    # io.imsave(save_dir + '/gt_s.png',gt_s )
    # io.imshow(gt)
    # io.show()


    # 预测专用
    # pred_u = sitk.GetArrayFromImage(sitk.ReadImage(pre_u_dir + '.nii.gz'))
    # pred_u,pred_u_s = show_segmented_image("/home/user/4TB/datasets/MICCAI_BraTS2020_TrainingData/"+case_number+"/"+case_number+'_flair.nii.gz', pre_u_dir + '.nii.gz')
    pred_u, pred_u_s = show_segmented_image(
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_flair.nii.gz',
        pre_u_dir + '.nii.gz')

    # pred_u = pred_u[z_number, :, :]
    pred_u = pred_u[ :, :,z_number]
    # pred_u = pred_u[:,z_number,  :]

    io.imsave(save_dir + '/L_pred_u.png', pred_u)
    # pred_u_s = pred_u_s[z_number, :, :]
    # pred_u_s = (pred_u_s * 255.0).astype('uint8')
    # io.imsave(save_dir + '/pred_u_s.png', pred_u_s)


    # pred_final = sitk.GetArrayFromImage(sitk.ReadImage(pre_final_dir + '.nii.gz'))
    # pred_final, pred_final_s = show_segmented_image("/home/user/4TB/datasets/MICCAI_BraTS2020_TrainingData/"+case_number+"/"+case_number+'_flair.nii.gz', pre_final_dir + '.nii.gz')
    pred_final, pred_final_s = show_segmented_image(
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_flair.nii.gz',
        pre_final_dir + '.nii.gz')

    # pred_final= pred_final[z_number, :, :]
    pred_final = pred_final[:, :,z_number]
    # pred_final = pred_final[:, z_number, :]

    io.imsave(save_dir + '/L_unet_pred_f.png', pred_final)
    # pred_final_s=pred_final_s[z_number, :, :]
    # pred_final_s = (pred_final_s * 255.0).astype('uint8')
    # io.imsave(save_dir + '/unet_pred_f_s.png', pred_final_s)
    # img_dir=sorted(glob(f"/home/user/4TB/Chenbonian/medic-segmention/case_images/case_{case_number}/L_*"))
    # print(img_dir)
    # for k in img_dir:
    #     seg_img = imageio.imread(k)
    #     seq = iaa.Sequential([
    #         # 说明一下Crop函数中的参数设置，其中percent(上，右，下，左)进行放大图像，keep_size代表是否保持原来的大小，False是不保留原有大小
    #         # 如果不使用Crop函数会导致切片之间间距太大，没有充满整个图片，脑肿瘤切片只是在整个图片的中间位置，看起来很难看
    #         iaa.Crop(percent=(0.17, 0.19, 0.13, 0.19), keep_size=False)])
    #     images_aug = seq.augment_images(seg_img)
    # # images_aug2 = seq.augment_images(images2)
    #
    #     io.imsave(save_dir + '/case_{}_seg.png'.format(case_number), np.hstack(images_aug))
    # io.imsave(save_dir + '/case_{}_2.png'.format(2), np.hstack(images_aug))
    # if save == True:
    #     case_1 = imageio.imread(output_dir + '/case_images/case_1/case_1.png')
    #     case_2 = imageio.imread(output_dir + '/case_images/case_2/case_2.png')
        # case_3 = imageio.imread(output_dir + '/case_images/case_3/case_3.png')
        # case_4 = imageio.imread(output_dir + './case_images/case_4/case_4.png')
        # case_5 = imageio.imread(output_dir + '/case_images/case_5/case_5.png')
        # images = [case_1, case_2, case_3, case_4, case_5]
        # print(case_1.shape, case_2.shape, case_3.shape, case_4.shape, case_5.shape)
        # io.imsave(output_dir + '/case_images/case_example.png', np.vstack(images))
from glob import glob
data = sorted(glob("/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/all/*.nii.gz"))
print(data)
# show_data = r'/BraTS20_Training_044'
for i in range(len(data)):
    fname = data[i].split("/")[-1].split(".")[0]
    fname = fname.split("_")[-1]
    print(fname)
    simple = r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2021_train/images/BraTS2021_'+fname
    simple_lab = r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2021_train/labels/BraTS2021_'+fname
    pre_dir_u = r'/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/all/BraTS21_'+fname
    pre_dir_final = r'/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/all/BraTS21_'+fname
    save_file = '/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/'

    save_visual_examples(simple,simple_lab,pre_dir_u,pre_dir_final,save_file, fname, 74, False)

# fname = 'BraTS21_00011'
# # simple = r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/images/'+fname
# simple = r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2021_train/images/BraTS2021_00002'
# # simple = r"/home/user/4TB/Chenbonian/"
# simple_lab = r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/labels/'+fname
# pre_dir_u = r'/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/'+fname
# pre_dir_final = r'/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/'+fname
# save_file = '/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/'
# save_visual_examples(simple,simple_lab,pre_dir_u,pre_dir_final,save_file, fname,150, False)
# print(unet_pre_dir)
# 显示图片
# image = imageio.imread('./output/case_images/case_1/case_1.png')
# plt.imshow(image)
# plt.show()

