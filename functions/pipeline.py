import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import gzip
import sys
import csv
import glob
import random
import sys
from torch import tensor
from multiprocessing import Pool
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAccuracy,
    BinaryJaccardIndex,
)
from functions.sam_functions import *
from functions.modified_predictor import modifiedPredictor
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class analyze:
    def __init__(self, num_class1:tuple[int,int], num_class2:tuple[int,int], num_class3:tuple[int,int]):
        self.num_class1: tuple[int, int] = num_class1
        self.num_class2: tuple[int, int] = num_class2
        self.num_class3: tuple[int, int] = num_class3
        self.metrics = {
            "dice_score": BinaryF1Score,
            "iou": BinaryJaccardIndex,
            "accuracy": BinaryAccuracy,
        }
        self.mp = modifiedPredictor()
        self.ground_truths = None
        self.batch_masks = None
        self.batch_size = None
    def generate_masks(self, embeddings, ground_truths, batch_size):
        """Generate masks for the given embeddings and ground_truths in a batch

        Args:
            embeddings (torch.tensor): (B1 x 256 x 64 x 64) embeddings in batch
            ground_truths (NDArray): (B1 x H x W) segmentation of ground truths in batch
            batch_size(int): number of prompts per image (B2)
        Returns:
            torch.tensor: Mask of the given embeddings (B1 x B2 x H x W) generated by SAM (B1 is the batch size of image,
            B2 is the batch size of prompts per image)
        """
        self.ground_truths = ground_truths
        batch_masks = []
        self.batch_size = batch_size
        for embedding, ground_truth in zip(embeddings,ground_truths):
            prompt_class_1 = batch_sample_from_class(batch_size, ground_truth, target_class = 1, n_foreground = self.num_class1[0], n_background=self.num_class1[1])
            prompt_class_2 = batch_sample_from_class(batch_size, ground_truth, target_class = 2, n_foreground = self.num_class2[0], n_background=self.num_class2[1])
            prompt_class_3 = batch_sample_from_class(batch_size, ground_truth, target_class = 3, n_foreground = self.num_class3[0], n_background=self.num_class3[1])

            logit_class_1 = self.mp(embedding,prompt_class_1[0].cuda(),prompt_class_1[1].cuda())
            logit_class_2 = self.mp(embedding,prompt_class_2[0].cuda(),prompt_class_2[1].cuda())
            logit_class_3 = self.mp(embedding,prompt_class_3[0].cuda(),prompt_class_3[1].cuda())

            logit_stack = torch.stack([logit_class_1,logit_class_2,logit_class_3],dim = 1)
            
            final_masks = multiclass_prob(logit_stack, hard_labels=True)
            batch_masks.append(final_masks)
        self.batch_masks = torch.stack((batch_masks), dim = 1)
        return self.batch_masks

    def scoring_function(self, f):
        """Generate scores for the predicted mask for each class

        Args:
            f(torch.classification.metric): scoring metrics for the mask and ground_truth

        Returns:
            torch.tensor: score of class 1,2,3 in batches (B1 x B2 x 3)
        """
        # Generate an empty tensor with 1 x C with C as number of classes
        self.gt_cuda = torch.as_tensor(self.ground_truths, dtype = torch.int, device = torch.device("cuda:0"))
        self.gt_cuda = self.gt_cuda().unsqueeze(1)
        self.gt_cuda = self.gt_cuda.repeat(1,self.batch_size,1,1)
        scores = torch.zeros([self.gt_cuda.shape[0],self.batch_size, 3], device= torch.device("cuda:0"))
        for c in [1, 2, 3] :
            pred = torch.where(self.batched_masks == c, 1, 0)
            target = torch.where(self.gt_cuda == c, 1, 0))
            metric = f()
            scores[:,:,c - 1] = metric(pred, target)
        return scores

    # def generate_score(batch_loader,metrics:list):
    # results = {}
    # for embedding,ground_truth in batch_loader:
    # multiprocessing.set_start_method("spawn")
    # p = Pool(4)
    # masks = p.map(self.generate_masks,zip(embedding,ground_truth)
    # for metric in metrics:
    # f = self.metrics[metric]
    # scores = p.map(self.scoring_function(f,zip(masks,ground_truth)))
    # if metric not in results.keys():
    # results[metric] = scores
    # else:
    # results[metric].append(scores)
    # return results


def gzip_file(file, mode, image=False):
    f = gzip.GzipFile(file, mode)
    if mode == "r":
        in_image = np.load(f)
        return in_image
    elif mode == "w":
        np.save(file, image)


class CustomData(Dataset):
    def __init__(self, path, debug=False):
        self.path = path
        self.get_data(debug)

    def get_data(self, debug=False):
        ground_truth = gzip_file(os.path.join(self.path, "ground_truth.npy.gz"), "r")
        embedding = torch.load(os.path.join(self.path, "embeddings.pt"))
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


def get_batch(path, batch_number, debug=False):
    dataset = CustomData(path, debug)
    data_loader = DataLoader(dataset, batch_size=batch_number, shuffle=False)
    return data_loader


class Results:
    def __init__(self, path_to_output_file, experiment_name):
        self.path = f"{path_to_output_file}/{experiment_name}.csv"
        self.column_names = [
            "id",
            "image_id",
            "Dataset",
            "f_points_class_1",
            "f_points_class_2",
            "f_points_class_3",
            "b_points_class_1",
            "b_points_class_2",
            "b_points_class_3",
            "dice",
            "IOU",
            "accuracy",
        ]  # think about column names, class scores?

        # Initialize the file with column names
        with open(self.path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.column_names)
            writer.writeheader()

    def append_row(self, rows):
        """
        Appends row of results to the results file

        Args:
            rows (list): List of results values.
            Should match the output of the analyzer class
        """
        with open(self.path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.column_names)
            writer.writerows(rows)
