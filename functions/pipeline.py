
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
from functions.modified_predictor import modifiedPredictor
import pandas as pd
from torch.utils.data import Dataset, DataLoader



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
        """Generate masks for the given embedding and ground_truth

        Args:
            embedding (torch.tensor): embedding of each image
            ground_truth (NDArray): segmentation of ground truth 

        Returns:
            NDArray: Mask of the given embedding (256 x 216) generated by SAM
        """
        predictor = modifiedPredictor(embedding,(1024, 864),(256,216))
        classes = [1,2,3]
        num_prompts = [self.num_class1,self.num_class2,self.num_class3]
        # Loop through the number of prompts for each class
        # Assign new number of prompts for this class if the number of pixels < number of prompts
        for idx, n in enumerate(num_prompts):
            if len(ground_truth[ground_truth == (idx + 1)])< n:
                num_prompts[idx] == len(ground_truth[ground_truth == (idx + 1)])
        masks = []
        # Generate a dictionary to store the prompts for each class (helpful when an image does not contain all 3 classes)
        prompts = {1:[],2:[],3:[]}
        for c in classes:
            if c in np.unique(ground_truth):
                prompts[c] = sample_from_class(ground_truth, c, num_prompts[c-1])
        for c in classes:
            if c in np.unique(ground_truth):
                # Generate the labels for prompts as inputs in sam predictor
                # Store the labels in a dictionary
                # To generate mask for a class, prompts for this class will be labeled as 1. Other prompts in other classes are labeled as 0
                labels = {1:[],2:[],3:[]}
                labels[c] = [1]*num_prompts[c-1]
                if self.negative_prompt:
                    # Input prompts for predictor are all prompts from all 3 classes, with foreground and background labels
                    input_points = np.concatenate(list(prompts.values()),axis = 0)
                    for nc in classes:
                        if nc != c:
                            labels[nc] = [0] * num_prompts[nc - 1]
                else:
                    # If only foreground, input prompts are only the class of interest 
                    input_points = prompts[c]
                prmt_labels = np.concatenate(list(labels.values()))
                mask, _, _ = predictor.predict(point_coords=input_points,\
                                                        point_labels=prmt_labels,\
                                                        multimask_output=False,\
                                                        return_logits=True)
            else:
                # If class is not present in an image, a logit score of -7 is generated for all masks
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

    #def generate_score(batch_loader,metrics:list):
        #results = {}
        #for embedding,ground_truth in batch_loader:
            #multiprocessing.set_start_method("spawn")
            #p = Pool(4)
            #masks = p.map(self.generate_masks,zip(embedding,ground_truth)
            #for metric in metrics:
                #f = self.metrics[metric]
                #scores = p.map(self.scoring_function(f,zip(masks,ground_truth)))
                #if metric not in results.keys():
                    #results[metric] = scores
                #else:
                    #results[metric].append(scores)
        #return results

    
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
        ground_truth = gzip_file(os.path.join(self.path,"ground_truth.npy.gz"),"r")
        embedding = torch.load(os.path.join(self.path,"embeddings.pt"))
        if debug:
            self.ground_truth = ground_truth[:50]
            self.embedding = embedding[:50]
        else:
            self.ground_truth = ground_truth
            self.embedding = embedding
    def __len__(self):
        return len(self.ground_truth)
    
    def __getitem__(self, idx):
        if torch.cuda.is_available():
            return self.embedding[idx].cuda(), self.ground_truth[idx]
        else:
            return self.embedding[idx], self.ground_truth[idx]
    
def get_batch(path,batch_number, debug = False):
    dataset = CustomData(path,debug)
    data_loader = DataLoader(dataset, batch_size = batch_number, shuffle=False)
    return data_loader



    


    

                
            
                            
                        
                    
                    
        
    


    

                
            
                            
                        
                    
                    
        