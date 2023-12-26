import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import gzip
import sys
import glob
import random
import sys
from torch import tensor
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import MulticlassAccuracy
from functions.preprocess import preprocess_image
from functions.sam_functions import get_logits,multiclass_prob,sample_from_class
import pandas as pd

os.chdir("..")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

def show_points(coords, labels, ax, marker_size=10):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='green', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='red', linewidth=1.25)   


def sam_input(gt_image, label: int, n_points: int, input_image = np.array([None]), pixel_values = range(256)):
    '''
    Generates an array of a defined number of sampled points from one class of the gt_image.

    Arguments:
    gt_image: 2D-array, assigned labels of the original image
    label: integer, class to which the sam_points should belong
    n_points: number of points that should be returned
    input_image: array of input image for checking pixel values
    pixel_values: a range of pixel values that sam_points should have
    Returns:
    A numpy array of n points belonging to the label
    '''
    if input_image.any() != None:
        minimum = min(pixel_values)
        maximum = max(pixel_values)
        arr = np.where(gt_image == label,input_image,0)
        rows,cols = np.where((minimum <= arr) & (maximum >= arr))
        points = [[col, row] for row, col in zip(rows,cols)]

    else:
        rows,cols = np.where(gt_image == label)
        points = [[col, row] for row, col in zip(rows,cols)]
    assert len(points) >= n_points, f"Choose the number of points lower than {len(points)}"
    sam_points = np.array(random.sample(points, n_points))
    return sam_points

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def generate_mask_mul_prts(image,gt_image,num_class_1,num_class_2,num_class_3):
    """Generate an array of mask for 3 classes in each Z-stack

    Args:
        image (array): Image array (HxWxZ dimension with Z as number of stacks)
        gt_image (array): Image array (HxWxZ dimension with Z as number of stacks)
        num_class_1 (int): Number of prompts for class 1
        num_class2 (int): Number of prompts for class 2
        num_class3 (int): Number of prompts for class 3

    Returns:
        Array: A mask array of dimension of H x W x Z(Z: Z-stacks)
    """
    mask_array = np.zeros(image.shape)
    for i in range(image.shape[2]):
        # Create RGB image for SAM input
        img = cv2.cvtColor(image[:,:,i],cv2.COLOR_GRAY2RGB)
        sam_img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        sam_img = sam_img.astype(np.uint8)
        # Use try because some slices do not contain segmentation
        classes = np.unique(gt_image[:,:,i])
        classes = classes[classes>0]
        if len(classes) == 3:
            # Generate points for 3 classes, class 2 is usually more difficult to recognize 
            points_1 = sam_input(gt_image[:,:,i],1,num_class_1)
            points_2 = sam_input(gt_image[:,:,i],2,num_class_2)
            points_3 = sam_input(gt_image[:,:,i],3,num_class_3)
            input_points = np.concatenate((points_1,points_2,points_3))
            # Generate predictor
            predictor = SamPredictor(sam)
            predictor.set_image(sam_img)
            masks = []
            for j in range(3):
                input_labels = np.zeros(num_class_1 + num_class_2 + num_class_3)
                if j == 0:
                    input_labels[:num_class_1] = 1
                elif j == 1:
                    input_labels[num_class_1 : (num_class_1 + num_class_2)] = 1
                elif j == 2:
                    input_labels[(num_class_1 + num_class_2) :] = 1
                mask, scores, logits = predictor.predict(point_coords=input_points,\
                                                        point_labels=input_labels,\
                                                        multimask_output=False,\
                                                        return_logits=True)
                masks.append(mask)
            mask_array[:,:,i] = multiclass_prob(masks, hard_labels=True)
    return mask_array

def generate_mask_sing_prts(image,gt_image,n_points):
    """Generate an array of mask for 3 classes in each Z-stack

    Args:
        image (array): Image array (HxWxZ dimension with Z as number of stacks)
        gt_image (array): Image array (HxWxZ dimension with Z as number of stacks)
        n_points: number of prompts for each class

    Returns:
        Array: A mask array of dimension of H x W x Z(Z: Z-stacks)
    """
    mask_array = np.zeros(image.shape)
    for i in range(image.shape[2]):
        img = cv2.cvtColor(image[:,:,i],cv2.COLOR_GRAY2RGB)
        sam_img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        sam_img = sam_img.astype(np.uint8)
        prompts = []
        predictor = SamPredictor(sam)
        predictor.set_image(sam_img)
        # Use if because some slices do not contain segmentation
        if len(np.unique(gt_image[:,:,i])) == 3 :
            for label in [1, 2, 3]:
                prompts.append(sample_from_class(gt_image[:,:,i], label, n_points=n_points))
            logits = get_logits(prompts, predictor)
            mask = multiclass_prob(logits, hard_labels=True)
            mask_array[:,:,i] = mask
    return mask_array

def Jaccard_score(masks, ground_truth):
    """Generate IoU score for the predicted mask for each class

    Args:
        masks (NDArray): H x W x Z mask containing annotation for class 0,1,2,3
        ground_truth (NDArray): H x W x Z ground containing annotation for class 0,1,2,3

    Returns:
        list: list of mean IoU score for class 0, 1, 2, 3 consecutively
    """
    # Generate an empty tensor with C x Z with C as number of classes, Z as number of stacks
    scores = torch.zeros([ground_truth.shape[2],4], device = "cuda:0")
    for i in range(ground_truth.shape[2]):
        pred = tensor(masks[:,:,i])
        target = tensor(ground_truth[:,:,i])
        metric = MulticlassJaccardIndex(num_classes=4, average = None)
        score = metric(pred, target)
        scores[i,:] = score
    # Filter for rows with scores > 0 for three masks
    filter_score = scores[(scores[:,1] > 0) & (scores[:,2] > 0) & (scores[:,3] > 0)]
    mean_scores = []
    for j in range(4):
        mean_score = round(float(filter_score[:,j].mean()),3)
        mean_scores.append(mean_score)
    return mean_scores

def Dice_score(masks, ground_truth):
    """Generate dice score for the predicted mask for each class

    Args:
        masks (NDArray): H x W x Z mask containing annotation for class 0,1,2,3
        ground_truth (NDArray): H x W x Z ground containing annotation for class 0,1,2,3

    Returns:
        list: list of dice score for class 0, 1, 2, 3 consecutively
    """
    # Generate an empty tensor with C x Z with C as number of classes, Z as number of stacks
    scores = torch.zeros([ground_truth.shape[2],4], device = "cuda:0")
    for i in range(ground_truth.shape[2]):
        pred = tensor(masks[:,:,i], dtype = int)
        target = tensor(ground_truth[:,:,i], dtype = int)
        metric = MulticlassF1Score(num_classes=4, average = None)
        score = metric(pred, target)
        scores[i,:] = score
    # Filter for rows with scores > 0 for three masks
    filter_score = scores[(scores[:,1] > 0) & (scores[:,2] > 0) & (scores[:,3] > 0)]
    mean_scores = []
    for j in range(4):
        mean_score = round(float(filter_score[:,j].mean()),3)
        mean_scores.append(mean_score)
    return mean_scores
    
def accuracy_score(masks, ground_truth):
    """Generate accuracy score for the predicted mask for each class

    Args:
        masks (NDArray): H x W x Z mask containing annotation for class 0,1,2,3
        ground_truth (NDArray): H x W x Z ground containing annotation for class 0,1,2,3

    Returns:
        list: list of mean accuracy score for class 0, 1, 2, 3 consecutively
    """
    # Generate an empty tensor with C x Z with C as number of classes, Z as number of stacks
    scores = torch.zeros([ground_truth.shape[2],4], device = "cuda:0")
    for i in range(ground_truth.shape[2]):
        pred = tensor(masks[:,:,i])
        target = tensor(ground_truth[:,:,i])
        mca = MulticlassAccuracy(num_classes=4, average=None)
        score = mca(pred, target)
        scores[i,:] = score
    # Filter for rows with scores > 0 for three masks
    filter_score = scores[(scores[:,1] > 0) & (scores[:,2] > 0) & (scores[:,3] > 0)]
    mean_scores = []
    for j in range(4):
        mean_score = round(float(filter_score[:,j].mean()),3)
        mean_scores.append(mean_score)
    return mean_scores
    
def gzip_file(file,mode,image = False):
    f = gzip.GzipFile(file,mode)
    if mode == "r":
        in_image = np.load(f)
        return in_image
    elif mode == "w":
        np.save(file,image)

def analyze():
    input_path = input("Input path:")
    output_path = input("Output path:")
    pipeline_name = input("Name of pipeline:")
    pipeline = input("Test of only positive prompts [y/n]:")
    if pipeline == "n":
        num_class_1 = int(input("Number of prompts for class 1:"))
        num_class_2 = int(input("Number of prompts for class 2:"))
        num_class_3 = int(input("Number of prompts for class 3:"))
    else: 
        num = int(input("Number of prompts per class:"))
    try:
        os.mkdir(output_path)
    except:
        pass
    out_folder = os.path.join(output_path,pipeline_name)
    try:
        os.mkdir(out_folder)
    except:
        pass
    os.chdir(input_path)
    dct = {"files":[]}
    for j in range(1,4):
        dct[f"accuracy_{j}"] = []
        dct[f"iou_{j}"] = []
        dct[f"dice_{j}"] = []
    for folder in os.listdir():
        input_folder = os.path.join(input_path,folder)
        os.chdir(input_folder)
        for file in glob.glob("*_gt.npy.gz"):
            print(file)
            gt_image = gzip_file(file,"r")
            image = gzip_file(file.replace("_gt",""),"r")
            dct["files"].append(file.replace("_gt",""))
            if pipeline == "n":
                masks = generate_mask_mul_prts(image,gt_image,num_class_1,num_class_2,num_class_3)
            else: 
                masks = generate_mask_sing_prts(image,gt_image,num)
            _,acc1,acc2,acc3 = accuracy_score(masks,gt_image)
            _,iou1,iou2,iou3 = Jaccard_score(masks,gt_image)
            _,dice1,dice2,dice3 = Dice_score(masks,gt_image)
            dct["accuracy_1"].append(acc1)
            dct["accuracy_2"].append(acc2)
            dct["accuracy_3"].append(acc3)
            dct["dice_1"].append(dice1)
            dct["dice_2"].append(dice2)
            dct["dice_3"].append(dice3)
            dct["iou_1"].append(iou1)
            dct["iou_2"].append(iou2)
            dct["iou_3"].append(iou3)
            gzip_file(os.path.join(out_folder,file.replace("_gt","")),"w",masks)
    df = pd.DataFrame(dct)
    df.to_csv(os.path.join(out_folder,"results.csv"))
    return df
                
            
                            
                        
                    
                    
        