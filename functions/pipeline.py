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
from multiprocessing import Pool
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy,BinaryJaccardIndex
from functions.sam_functions import get_logits,multiclass_prob,sample_from_class
from functions.generate_embeddings import modifiedPredictor
import pandas as pd
from torch.utils.data import Dataset, DataLoader

os.chdir("..")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)


class analyze:
    def __init__(self,num_class1,num_class2,num_class3, negative_prompt = False):
        self.num_class1 = num_class1
        self.num_class2 = num_class2
        self.num_class3 = num_class3
        self.negative_prompt = negative_prompt
        self.metrics = {
            "dice_score":BinaryF1Score,
            "iou":BinaryJaccardIndex,
            "accuracy":BinaryAccuracy

            }

    def generate_masks(self,embedding,ground_truth):
        predictor = modifiedPredictor(embedding,(1024, 864),(256,216))
        classes = [1,2,3]
        num_prompts = [self.num_class1,self.num_class2,self.num_class3]
        for idx, n in enumerate(num_prompts):
            if len(ground_truth[ground_truth == (idx + 1)])< n:
                num_prompts[idx] == len(ground_truth[ground_truth == (idx + 1)])
        masks = []
        prompts = {1:[],2:[],3:[]}
        for c in classes:
            if c in np.unique(ground_truth):
                prompts[c] = sample_from_class(ground_truth, c, num_prompts[c-1])
        for c in classes:
            if c in np.unique(ground_truth):
                labels = {1:[],2:[],3:[]}
                labels[c] = [1]*num_prompts[c-1]
                if self.negative_prompt:
                    input_points = np.concatenate(list(prompts.values()),axis = 0)
                    for nc in classes:
                        if nc != c:
                            labels[nc] = [0] * num_prompts[nc - 1]
                else:
                    input_points = prompts[c]
                prmt_labels = np.concatenate(list(labels.values()))
                print(prmt_labels)
                mask, _, _ = predictor.predict(point_coords=input_points,\
                                                        point_labels=prmt_labels,\
                                                        multimask_output=False,\
                                                        return_logits=True)
            else:
                mask = np.fill((256,216),-7)
            masks.append(mask)
        final_mask = multiclass_prob(masks, hard_labels=True)
        return final_mask

    def scoring_function(self,f,mask, ground_truth):
        """Generate scores for the predicted mask for each class

        Args:
            f(torch.classification.metric): scoring metrics for the mask and ground_truth
            masks (NDArray): H x W x Z mask containing annotation for class 0,1,2,3
            ground_truth (NDArray): H x W x Z ground containing annotation for class 0,1,2,3

        Returns:
            list: list of mean IoU score for class 0, 1, 2, 3 consecutively
        """
        # Generate an empty tensor with 1 x C with C as number of classes
        scores = torch.zeros([1,3], device = "cuda:0")
        for c in range[1,2,3]:
            pred = tensor(np.where(mask == c,1,0))
            target = tensor(np.where(ground_truth == c,1,0))
            metric = f()
            scores[:,c-1] = metric(pred,target)
        return scores

    def generate_score(batch_loader,metrics:list):
        results = {}
        for embedding, ground_truth in batch_loader:
            p = Pool(4)
            masks = p.map(self.generate_masks,embedding,ground_truth)
            for metric in metrics:
                f = self.metrics[metric]
                scores = p.map(self.scoring_function(f,masks,ground_truth))
                if metric not in results.keys():
                    results[metric] = scores
                else:
                    results[metric].append(scores)
        return results



    
def gzip_file(file,mode,image = False):
    f = gzip.GzipFile(file,mode)
    if mode == "r":
        in_image = np.load(f)
        return in_image
    elif mode == "w":
        np.save(file,image)

class CustomData(Dataset):
    def __init__(self, path, debug = False):
        self.path = path
        self.get_data(debug)
    
    def get_data(self,debug = False):
        data = gzip_file(os.path.join(self.path,"ground_truth.npy.gz"),"r")
        if debug:
            self.data = data[:50]
        else:
            self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def get_batch(path,batch_number):
    dataset = CustomData(path)
    data_loader = DataLoader(dataset, batch_size = batch_number, shuffle=False)
    return data_loader



    


    

                
            
                            
                        
                    
                    
        