from keras.models import load_model
import nibabel as nib

def load_image(path):
    img_load = nib.load(path)
    n = img_load.get_data().astype(np.float32)
    affine = img_load.affine

    return img_load, affine

x, affine = load_image('')
model = load_model('model.h5')
segment = model.predict(x)

img_new = nib.Nifti1Image(segment, affine)
nib.save(img_new, 'predicted_segmentation')