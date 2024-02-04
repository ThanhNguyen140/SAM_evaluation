from skimage.transform import resize
import os
from multiprocessing import pool
import numpy as np
import nibabel as nib
import sys
import glob
import gzip
import gzip
import torch

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

def remove_no_seg(path:str):
    """ Remove slices containing no segmentations. Note: This function will overwrite the original files
    Args:
    path (str): path to ground truth embeddings files
    Return:
    torch.Tensor: new embeddings with desired list of indices
    NDArray: new ground truth with slices containing segmentations """
    f = gzip.GzipFile(f"{path}/ground_truth.npy.gz","r")
    ground_truths = np.load(f)
    embeddings = torch.load(f"{path}/embeddings.pt")
    all_zeros = np.all(gt == 0, axis = 1)
    # Set indices contain no segmentations to True. Images with segmentations have False
    non_seg = np.all(all_zeros == True, axis = 1)
    # Get list of indices where segmentations are present
    ind = np.where(non_seg == False)
    new_ground_truths = ground_truths[ind]
    new_embeddings = embeddings[ind]
    f2 = gzip.GzipFile(f"{path}/ground_truth.npy.gz","w")
    np.save(f2,new_ground_truths)
    torch.save(new_embeddings,f"{path}/embeddings.pt")
    return new_embeddings,new_ground_truths

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
