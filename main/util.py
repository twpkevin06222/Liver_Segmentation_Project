import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib

#Parameter
NEW_SIZE = 128

def load_images(path):
    main_dir = sorted(os.listdir(path)) #file directory 

    img_list = []
    img_affine = []
    h = []
    s = []

    for dir in main_dir:
        data_path = os.path.join(path,dir)
        img_load = nib.load(data_path)
        n = img_load.get_data().astype(np.float32)
        affine = img_load.affine
        img_list.append(n) #image info stored in a list
        img_affine.append(affine)
        h.append(img_load.shape[1]) #since width and height are the same, only height is listed
        s.append(img_load.shape[2]) #slices

    return img_list, h, s, img_affine


def show_histograms(h, s):
    plt.hist(h, edgecolor = 'black', linewidth = 1.2)
    plt.xlabel('Width/Height')
    plt.ylabel('Amount')
    plt.show()

    plt.hist(s, color = 'orange', edgecolor = 'black', linewidth = 1.2)
    plt.xlabel('Slices')
    plt.ylabel('Amount')
    plt.show()

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


def save(img_list, img_affine, output_path, label='FALSE'):
    for i in range(len(img_list)):
        img_new = nib.Nifti1Image(img_list[i], img_affine[i])
        if label=='FALSE':
            nib.save(img_new, output_path+'/%d.nii.gz' %(i))
        else:
            nib.save(img_new, output_path+'/label_%d.nii.gz' %(i))
    pass

def three_to_two(path):
    ff = sorted(glob.glob(path))
    images =[]
    for f in range(len(ff)):
        a = nib.load(ff[f])
        a = a.get_data()
        for i in range(a.shape[2]):
            images.append(a[:,:,i])     
    images = np.asarray(images)
    return images

def min_max_norm(images):
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi)/ (m - mi)
    return images

def label_outliers(img_labels):
    img_labels[img_labels>1]=1
    img_labels[img_labels<0]=0
    return img_labels

def data_process(input_path, output_path,npy_name,label='FALSE'):
    img_list, height, depth, img_affine = load_images(input_path)
    img_list = resize(img_list)
    os.mkdir(output_path)  
    new_img_list = save(img_list, img_affine, output_path, label)
    images = three_to_two(path = output_path + '/*')
    
    if label=='FALSE':
        img = min_max_norm(images)
        np.save(npy_name+'.npy', img)
    else:
        img_lbl = label_outliers(images)
        np.save(npy_name+'.npy', img_lbl)
    pass

#Input Path
input_path = {}
input_path['train'] = "/home/kevinteng/Desktop/Medical Imaging/data/train/img"
input_path['train_label'] = "/home/kevinteng/Desktop/Medical Imaging/data/train/seg"
input_path['test'] = "/home/kevinteng/Desktop/Medical Imaging/data/test/img"
input_path['test_label'] = "/home/kevinteng/Desktop/Medical Imaging/data/test/seg"

#Output Path
output_path = {}
output_path['train'] = 'img_new'
output_path['train_label'] = 'img_new_label'
output_path['test'] = 'test_img'
output_path['test_label'] = 'test_img_label'


#Main
data_process(input_path['train'], output_path['train'], 'x_data', label='FALSE')
data_process(input_path['train_label'], output_path['train_label'], 'y_data', label='TRUE')
data_process(input_path['test'], output_path['test'], 'x_test', label='FALSE')
data_process(input_path['test_label'], output_path['test_label'], 'y_test', label='TRUE')







