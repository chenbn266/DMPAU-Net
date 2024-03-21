# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import nibabel
import numpy as np
from tqdm import tqdm

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument("--preds", type=str, default="/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_last-v27_task=23_fold=0_tta", help="pred")
# parser.add_argument("--lbls", type=str, default="/home/user/4TB/Chenbonian/medic-segmention/isnet/results/last-v27-1000-500-0.7-0.3-tta",help="Path to labels")
parser.add_argument("--preds", type=str, default="/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_epoch=319-dice=89_66_task=19_fold=0_tta", help="pred")
parser.add_argument("--lbls", type=str, default="/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/labels",help="Path to labels")
# parser.add_argument("--preds", type=str, default="/home/user/4TB/Chenbonian/medic-segmention/isnet/results/new-bast-last-v33-1000-500-0.7-0.3-0.4-tta-2023", help="pred")

def get_stats(pred, targ, class_idx):
    tp = np.logical_and(pred == class_idx, targ == class_idx).sum()
    fn = np.logical_and(pred != class_idx, targ == class_idx).sum()
    fp = np.logical_and(pred == class_idx, targ != class_idx).sum()
    return tp, fn, fp

# def get_HD95(pred,targ,class_idx):
#
#     return

if __name__ == "__main__":
    args = parser.parse_args()
    y_pred = sorted(glob.glob(os.path.join(args.preds, "*.npy")))
    # y_pred = y_pred[:10]
    # y_true = [os.path.join(args.lbls, os.path.basename(pred).replace("npy", "nii.gz")) for pred in y_pred]
    y_true=[]
    for pred in y_pred:
        pred = pred.split("/")[-1]
        # print(pred)
        name = pred.split("_")
        print(name)
        # y_t = os.path.join(args.lbls, name[0]+"_seg.nii.gz")
        num = name[-1].split('.')
        y_t = os.path.join(args.lbls, "BraTS20_Training_" +num[0]+ "_seg.nii.gz")
        print(y_t)
        y_true.append(y_t)


    assert len(y_pred) > 0
    # n_class = np.load(y_pred[0]).shape[0] - 1
    n_class  =3

    dice = [[] for _ in range(n_class)]
    for pr, lb in tqdm(zip(y_pred, y_true), total=len(y_pred)):
        prd = np.transpose(np.argmax(np.load(pr), axis=0), (2, 1, 0))
        # prd = nibabel.load(pr).get_fdata().astype(np.uint8)
        lbl = nibabel.load(lb).get_fdata().astype(np.uint8)

        for i in range(1, n_class + 1):
            counts = np.count_nonzero(lbl == i) + np.count_nonzero(prd == i)
            if counts == 0:  # no foreground class
                dice[i - 1].append(1)
            else:
                tp, fn, fp = get_stats(prd, lbl, i)
                denum = 2 * tp + fp + fn
                dice[i - 1].append(2 * tp / denum if denum != 0 else 0)

    dice_score = np.mean(np.array(dice), axis=-1)
    dice_cls = " ".join([f"L{i+1} {round(dice_score[i], 4)}" for i, dice in enumerate(dice_score)])
    print(f"mean dice: {round(np.mean(dice_score), 4)} - {dice_cls}")
