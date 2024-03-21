import json
import os
from glob import glob
from subprocess import call
import time

import nibabel
import numpy as np
from joblib import Parallel, delayed



data = "/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/images/BraTS20_Training_001.nii.gz"

# data = sorted(glob(os.path.join(data ,"BraTS*")))
a =nibabel.load(data)
print(a.shape)