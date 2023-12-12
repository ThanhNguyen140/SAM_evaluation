from skimage.transform import resize
import os
from multiprocessing import pool
import numpy as np
import nibabel as nib
import sys
import glob
import gzip


def resize_image(image, old_spacing, new_spacing, order=3):  # -> Any:
    new_shape = (
        int(np.round(old_spacing[0] / new_spacing[0] * float(image.shape[0]))),
        int(np.round(old_spacing[1] / new_spacing[1] * float(image.shape[1]))),
        int(np.round(old_spacing[2] / new_spacing[2] * float(image.shape[2]))),
    )
    return resize(image, new_shape, order=order, mode="edge")


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def preprocess_image(
    nib_image, spacing_target=(10, 1.25, 1.25), is_seg=False, keep_z_spacing=False
):
    spacing = np.array(nib_image.header.get_zooms())[[2, 1, 0]]
    image = nib_image.get_fdata()
    if keep_z_spacing:
        spacing_target = list(spacing_target)
        spacing_target[0] = spacing[0]
    if not is_seg:
        order_img = 3
        if not keep_z_spacing:
            order_img = 1
        image = resize_image(image, spacing, spacing_target, order=order_img).astype(
            np.float32
        )
        image -= image.mean()
        image /= image.std()
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            results.append(
                resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None]
            )
        image = vals[np.vstack(results).argmax(0)]
    return image


if __name__ == "__main__":
    path = sys.argv[1]
    output_dir = os.path.join(path, "preprocess")
    try:
        os.mkdir(output_dir)
    except:
        pass
    for folder in os.listdir(path):
        sub_dir = os.path.join(path, folder)
        if not os.path.isfile(sub_dir) and folder != "preprocess":
            for sub_folder in os.listdir(sub_dir):
                input_folder = os.path.join(sub_dir, sub_folder)
                if not os.path.isfile(input_folder):
                    output_folder = os.path.join(output_dir, sub_folder)
                    try:
                        os.mkdir(output_folder)
                    except:
                        pass
                    os.chdir(input_folder)
                    for file in glob.glob("*.nii.gz"):
                        if "gt" in file:
                            image = nib.load(file)
                            pre_image = preprocess_image(
                                image, is_seg=True, keep_z_spacing=True
                            )
                            output = os.path.join(output_folder, file)
                            f = gzip.GzipFile(output.replace(".nii.gz", ".npy.gz"), "w")
                            np.save(f, pre_image)
                        elif ("frame" in file) and ("gt" not in file):
                            image = nib.load(file)
                            pre_image = preprocess_image(
                                image, is_seg=False, keep_z_spacing=True
                            )
                            output = os.path.join(output_folder, file)
                            f = gzip.GzipFile(output.replace(".nii.gz", ".npy.gz"), "w")
                            np.save(f, pre_image)
