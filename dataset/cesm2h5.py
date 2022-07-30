import os
import glob
import h5py
#import scipy.misc as misc
import numpy as np
import imageio
import sys

dataset_dir = "/home/jinyang.liu/lossycompression/cesm-multisnapshot-5fields"
field = sys.argv[1]

f = h5py.File("CESM_{}.h5".format(field), "w")
dt = h5py.special_dtype(vlen=np.dtype(np.float32))

for subdir in ["HR", "X2"]:
    if subdir in ["HR"]:
        im_paths = glob.glob(os.path.join(dataset_dir,field,
                                          "hr_train", 
                                          "*.dat"))

    else:
        im_paths = glob.glob(os.path.join(dataset_dir,field,
                                          "lr_train", 
                                          "*.dat"))
    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = np.fromfile(path,dtype=np.float32).reshape((1800,3600,1))
        #print(np.max(im))
        #print(np.min(im))
        print(path)
        grp.create_dataset(str(i), data=im)
