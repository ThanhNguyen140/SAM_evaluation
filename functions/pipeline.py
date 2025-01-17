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
    def __init__(self, embeddings, ground_truths):
        """_summary_

        Args:
            embeddings (torch.tensor): (B1 x 256 x 64 x 64) embeddings in batch
            ground_truths (NDArray): (B1 x H x W) segmentation of ground truths in batch
        """

        self.metrics = {
            "dice_score": BinaryF1Score,
            "iou": BinaryJaccardIndex,
            "accuracy": BinaryAccuracy,
        }
        self.mp = modifiedPredictor()
        self.ground_truths = ground_truths
        self.embeddings = embeddings
        self.batch_masks = None
        self.batch_size = None

    def generate_masks(
        self, prompt_class_1, prompt_class_2, prompt_class_3, batch_size_of_prompts
    ):
        """Generate masks for the given embeddings and ground_truths in a batch

        Args:
            batch_size_of_prompts(int): number of prompts per image (B2)
            prompt_class_1 (list[tuple]): list with prompts for B1 number of images for class 1. Each tuple contains prompts and labels
            prompt_class_2 (list[tuple]): list with prompts for B1 number of images for class 2. Each tuple contains prompts and labels
            prompt_class_3 (list[tuple]): list with prompts for B1 number of images for class 3. Each tuple contains prompts and labels
        Returns:
            torch.tensor: Mask of the given embeddings (B1 x B2 x H x W) generated by SAM (B1 is the batch size of image,
            B2 is the batch size of prompts per image)
        """
        batch_masks = []
        self.batch_size = batch_size_of_prompts
        self.prompt_class_1 = prompt_class_1
        self.prompt_class_2 = prompt_class_2
        self.prompt_class_3 = prompt_class_3
        for embedding, pr1, pr2, pr3 in zip(
            self.embeddings,
            self.prompt_class_1,
            self.prompt_class_2,
            self.prompt_class_3,
        ):
            if list(pr1[1].unique()) != [tensor(0)]:
                logit_class_1 = self.mp.predict(embedding, pr1[0].cuda(), pr1[1].cuda())
            else:
                logit_class_1 = torch.full((self.batch_size, 1, 256, 216), -7).cuda()
            if list(pr2[1].unique()) != [tensor(0)]:
                logit_class_2 = self.mp.predict(embedding, pr2[0].cuda(), pr2[1].cuda())
            else:
                logit_class_2 = torch.full((self.batch_size, 1, 256, 216), -7).cuda()
            if list(pr3[1].unique()) != [tensor(0)]:
                logit_class_3 = self.mp.predict(embedding, pr3[0].cuda(), pr3[1].cuda())
            else:
                logit_class_3 = torch.full((self.batch_size, 1, 256, 216), -7).cuda()
            logit_stack = torch.cat(
                [logit_class_1, logit_class_2, logit_class_3], dim=1
            )

            final_masks = multiclass_prob_batched(logit_stack, hard_labels=True)
            batch_masks.append(final_masks)
            del logit_class_1
            del logit_class_2
            del logit_class_3
        batch_masks = torch.stack(batch_masks, dim=0)
        self.batch_masks = batch_masks[:, :, 0, :, :]
        del self.embeddings
        return self.batch_masks.cpu()

    def scoring_function(self, f):
        """Generate scores for the predicted mask for each class

        Args:
            f(torch.classification.metric): scoring metrics for the mask and ground_truth

        Returns:
            torch.tensor: score of class 1,2,3 in batches (B1 x B2 x 3)
        """
        # Generate an empty tensor with 1 x C with C as number of classes
        self.gt_cuda = torch.as_tensor(
            self.ground_truths, dtype=torch.int, device=torch.device("cuda:0")
        )
        self.gt_cuda = torch.unsqueeze(self.gt_cuda, dim=1)
        self.gt_cuda = self.gt_cuda.repeat(1, self.batch_size, 1, 1)
        scores = torch.zeros(
            [self.gt_cuda.shape[0], self.batch_size, 3], device=torch.device("cuda:0")
        )
        for c in [1, 2, 3]:
            preds = torch.where(self.batch_masks.cuda() == c, 1, 0)
            targets = torch.where(self.gt_cuda == c, 1, 0)
            metric = f().to(torch.device("cuda:0"))
            for idx in range(preds.shape[0]):
                for b in range(preds.shape[1]):
                    scores[idx, b, c - 1] = metric(
                        preds[idx, b, :, :], targets[idx, b, :, :]
                    )
        self.gt_cuda = self.gt_cuda.cpu()
        self.batch_masks = self.batch_masks.cpu()
        return scores.cpu()

    def get_results(
        self,
        dice_scores: torch.Tensor,
        iou_scores: torch.Tensor,
        accuracy_scores: torch.Tensor,
        num_prompt_class1: tuple[int, int],
        num_prompt_class2: tuple[int, int],
        num_prompt_class3: tuple[int, int],
    ):
        """

        Args:
            dice_scores (torch.Tensor): _description_
            iou_scores (torch.Tensor): _description_
            accuracy_scores (torch.Tensor): _description_
            num_prompt_class1 (tuple[int,int]): _description_
            num_prompt_class2 (tuple[int,int]): _description_
            num_prompt_class3 (tuple[int,int]): _description_
        """
        results = []
        for idx in range(dice_scores.shape[0]):
            for b in range(dice_scores.shape[1]):
                result = {
                    "image_id": idx,
                    "f_points_class_1": num_prompt_class1[0],
                    "f_points_class_2": num_prompt_class2[0],
                    "f_points_class_3": num_prompt_class3[0],
                    "b_points_class_1": num_prompt_class1[1],
                    "b_points_class_2": num_prompt_class2[1],
                    "b_points_class_3": num_prompt_class3[1],
                    "dice_class_1": round(float(dice_scores[idx, b, 0]), 3),
                    "dice_class_2": round(float(dice_scores[idx, b, 1]), 3),
                    "dice_class_3": round(float(dice_scores[idx, b, 2]), 3),
                    "IOU_class_1": round(float(iou_scores[idx, b, 0]), 3),
                    "IOU_class_2": round(float(iou_scores[idx, b, 1]), 3),
                    "IOU_class_3": round(float(iou_scores[idx, b, 2]), 3),
                    "accuracy_class_1": round(float(accuracy_scores[idx, b, 0]), 3),
                    "accuracy_class_2": round(float(accuracy_scores[idx, b, 1]), 3),
                    "accuracy_class_3": round(float(accuracy_scores[idx, b, 2]), 3),
                }
                results.append(result)
        return results


def gzip_file(file, mode, image=False):
    f = gzip.GzipFile(file, mode)
    if mode == "r":
        in_image = np.load(f)
        return in_image
    elif mode == "w":
        np.save(f, image)


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
            "image_id",
            "f_points_class_1",
            "f_points_class_2",
            "f_points_class_3",
            "b_points_class_1",
            "b_points_class_2",
            "b_points_class_3",
            "dice_class_1",
            "dice_class_2",
            "dice_class_3",
            "IOU_class_1",
            "IOU_class_2",
            "IOU_class_3",
            "accuracy_class_1",
            "accuracy_class_2",
            "accuracy_class_3",
        ]  # think about column names, class scores?

        # Initialize the file with column names if not already done
        if not os.path.exists(self.path):
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

def get_class_list(class_num: int,ground_truth) -> list[int]:
    """Get list of indices for each class for result analysis

    Args:
        class_num (int): a class of interest among classes 1,2,3
        ground_truth (NDArray): ground truth arrays where segmentations of each class are 
        annotated

    Returns:
        list: list of indices containing the class of interest
    """
    array1 = np.any(ground_truth == class_num, axis = 1)
    array2 = np.any(array1 == True, axis = 1)
    class_indices = np.where(array2 == True)
    indices = list(class_indices[0])
    return indices