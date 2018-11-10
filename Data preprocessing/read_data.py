import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def load_images(path):
    main_dir = sorted(os.listdir(path)) #file directory 

    img_list = []
    h = []
    s = []

    for dir in main_dir:
        data_path = os.path.join(path,dir)
        print(data_path)
        img_load = nib.load(data_path)
        n = img_load.get_data().astype(np.float32)
        img_list.append(n) #image info stored in a list
        h.append(img_load.shape[1]) #since width and height are the same, only height is listed
        s.append(img_load.shape[2]) #slices

    return img_list, h, s


def show_histograms(h, s):
    plt.hist(h, edgecolor = 'black', linewidth = 1.2)
    plt.xlabel('Width/Height')
    plt.ylabel('Amount')
    plt.show()

    plt.hist(s, color = 'orange', edgecolor = 'black', linewidth = 1.2)
    plt.xlabel('Slices')
    plt.ylabel('Amount')
    plt.show()
