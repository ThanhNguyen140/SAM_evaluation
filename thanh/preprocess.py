from skimage.transform import resize
import os
from multiprocessing import pool
import numpy as np
import nibabel as nib
import sys
import glob
import gzip


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def preprocess_image(nib_image, new_size=(216, 256,0), is_seg=False, keep_z_spacing=False):
    
    image = nib_image.get_fdata()
    new_size = list(new_size)
    if keep_z_spacing:
        new_size[-1] = image.shape[-1]
    if not is_seg:
        order_img = 3
        if not keep_z_spacing:
            order_img = 1
        image = resize(image, new_size, order=order_img, mode = "edge").astype(
            np.float32)
        image -= image.mean()
        image /= image.std()
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            results.append(
                resize(tmp[i].astype(float), new_size, 1, mode = "edge")[None]
            )
        image = vals[np.vstack(results).argmax(0)]
    new_image = image.transpose()
    return new_image


if __name__ == "__main__":
    path = sys.argv[1]
    output_dir = os.path.join(path, "preprocess")
    try:
        os.mkdir(output_dir)
    except:
        pass
    ground_truth = []
    images = []
    for folder in os.listdir(path):
        sub_dir = os.path.join(path, folder)
        if not os.path.isfile(sub_dir) and folder != "preprocess":
            for sub_folder in os.listdir(sub_dir):
                input_folder = os.path.join(sub_dir, sub_folder)
                if not os.path.isfile(input_folder):
                    os.chdir(input_folder)
                    for file in glob.glob("*.nii.gz"):
                        if "gt" in file:
                            gt_image = nib.load(file)
                            gt_pre_image = preprocess_image(
                                gt_image, is_seg=True, keep_z_spacing=True
                            )
                            ground_truth.append(gt_pre_image)
                            image_file = file.replace("_gt","")
                            image = nib.load(image_file)
                            pre_image = preprocess_image(
                                image, is_seg=False, keep_z_spacing=True
                            )
                            images.append(pre_image)

    gt_output = os.path.join(output_dir,"ground_truth.npy.gz")
    gt_f = gzip.GzipFile(gt_output, "w")
    ground_truth = np.vstack(ground_truth)
    np.save(gt_f, ground_truth)
    images = np.vstack(images)
    output = os.path.join(output_dir,"images.npy.gz")
    f = gzip.GzipFile(output, "w")
    np.save(f, images)
