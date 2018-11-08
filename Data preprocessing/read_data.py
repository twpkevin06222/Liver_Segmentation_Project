import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = "/home/kevinteng/Desktop/Medical Imaging/data/train/img" #directory to 'img' folder
main_dir = sorted(os.listdir(path)) #file directory 

img_list = []
h = []
s = []

for dir in main_dir:
    data_path = os.path.join(path,dir)
    img_load = nib.load(data_path)
    n = img_load.get_data().astype(np.float32)
    img_list.append(n) #image info stored in a list
    h.append(img_load.shape[1]) #since weight and height are the same, only height is listed
    s.append(img_load.shape[2]) #slices

plt.hist(h, edgecolor = 'black', linewidth = 1.2)
plt.xlabel('Width/Height')
plt.ylabel('Amount')
plt.show()

plt.hist(s, color = 'orange', edgecolor = 'black', linewidth = 1.2)
plt.xlabel('Slices')
plt.ylabel('Amount')
plt.show()
