from os.path import basename, join, exists
from skimage import io
import sys

import os
dir_ = sys.argv[1]
file_list = sorted(os.listdir(dir_),key=lambda x: int(basename(x).split('.')[0]))
print("reading files")
data = []
for i in file_list:
    data.append(io.imread(join(dir_,i)))
    
import numpy as np
images = np.stack(data)

mm = np.mean(images,axis=0)
print("doing svd")
imgmat = (images - mm).reshape(415,-1)
U, sigma, V = np.linalg.svd(imgmat,full_matrices=False)


x = sys.argv[2]

print("starting reconstructing {}".format(x))
i = int(basename(x).split('.')[0])
xx = np.dot(U[i,:4], np.dot(np.diag(sigma[:4]), V[:4,:]))
xx = xx.reshape(600,600,3)
xx += mm
io.imsave('reconstruction.jpg',xx.clip(0,255).astype(np.uint8))



