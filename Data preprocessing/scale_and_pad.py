from read_data import load_images
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib

NEW_SIZE = 128

def resize(img_list):
    new_img_list = []

    for img in img_list:
        new_img = []

        for i in range(np.size(img, 2)):
            new_img.append(np.asarray(Image.fromarray(img[:,:,i],mode='F').resize((NEW_SIZE,NEW_SIZE), Image.LANCZOS)))

        new_img_list.append(np.swapaxes(new_img, 0, 2))

    return new_img_list

def depth_pad(img_list, max_depth):
    img_list_pad = []
    for i in range(len(img_list)):
        get_depth = img_list[i].shape[2]
        if get_depth != max_depth:
            pad_layer = max_depth - get_depth
            padding = np.zeros((NEW_SIZE, NEW_SIZE, pad_layer))
            img_pad = np.dstack([padding, img_list[i]]) # stacking zero padding on top on available images
            img_list_pad.append(img_pad)
        else: 
            img_list_pad.append(img_list[i])
    return img_list_pad


def save(img_list, img_affine):
    for i in range(len(img_list)):
        img_new = nib.Nifti1Image(img_list[i], img_affine[i])
        nib.save(img_new, '<out path>/img_new/%d.nii.gz' %(i))
    pass

img_list, height, depth, img_affine = load_images("<input path>/data/train/img") 
img_list = resize(img_list)
img_list_pad = depth_pad(img_list, max_depth = max(depth))
new_img_list = save(img_list_pad, img_affine) 
