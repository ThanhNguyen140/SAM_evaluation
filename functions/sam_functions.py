import os
import pickle
import numpy as np
import torch
import random
from segment_anything import SamPredictor, sam_model_registry
from functions.preprocess import preprocess_image


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=10):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="o", s=marker_size
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="o", s=marker_size
    )


def sample_from_class(ground_truth, target_class: int, n_points: int):
    """
    Generates a list of randomly sampled points from one class of the ground_truth and returns
    them in a readable form for SAM.

    Arguments:
    ground_truth: 2D-array, assigned labels of the original image
    target_class: integer, class to which the sampled points should belong
    n_points: number of points that should be returned

    Returns:
    list of n points belonging to the target_class
    """
    rows, cols = np.where(ground_truth == target_class)
    points = [[col, row] for row, col in zip(rows, cols)]
    sampled_points = np.array(random.sample(points, n_points))
    return sampled_points


def batch_sample_from_class(
    batch_size, ground_truth, target_class: int, n_foreground: int, n_background=0
):
    """
    Randomly sample n points that belong to the target_class for one batch.

    Arguments:
    batch_size: integer, number of prompts that should be created
    ground_truth: 2D-array (H,W), assigned labels of the original image
    target_class: integer, class to which the sampled points should belong
    n_foreground: number of points that should be sampled from the foreground
    n_background: number of points that should be sampled from the background (default=0)

    Returns:
    [Tuple[torch.Tensor, torch.Tensor]], list length: batch_size, tensor length: n_points
    [points, labels]
    """
    batched_points = []
    batched_labels = []
    class_indices = torch.nonzero(ground_truth == target_class, as_tuple=False)
    other_class_indices = torch.nonzero(
        (ground_truth != target_class) & (ground_truth != 0), as_tuple=False
    )

    if len(class_indices) < n_foreground:  # class has not as many pixels
        n_foreground = len(class_indices)  # change number of points to sample
        print(f"n_foreground was reduced to {n_foreground}.")

    if len(other_class_indices) < n_background:  # class has not as many pixels
        n_background = len(other_class_indices)  # change number of points to sample
        print(f"n_background was reduced to {n_background}.")

    for _ in range(batch_size):
        points = class_indices[
            np.random.choice(class_indices.shape[0], n_foreground, replace=False)
        ]
        points[:, [0, 1]] = points[:, [1, 0]]  # swap axes
        labels = torch.tensor([1] * len(points))
        if n_background > 0:
            background_points = other_class_indices[
                np.random.choice(
                    other_class_indices.shape[0], n_background, replace=False
                )
            ]
            background_points[:, [0, 1]] = background_points[:, [1, 0]]  # swap axes
            background_labels = torch.tensor([0] * len(background_points))
            points = torch.cat([points, background_points], dim=0)
            labels = torch.cat([labels, background_labels], dim=0)

        batched_points.append(points)
        batched_labels.append(labels)

    # Convert lists to tensors
    batched_points = torch.stack(batched_points)
    batched_labels = torch.stack(batched_labels)

    return batched_points, batched_labels


def get_masks(prompts, predictor):
    """This function returns the models masks for a given list of prompts.

    Args:
        prompts (list): List of lists of sampled points from ground truth
        predictor (SAMPredictor): Predictor that has already a set image

    Returns:
        np.array: 3D array with the masks (one for each class)
    """
    masks = []
    for label, prompt in enumerate(prompts):
        input_labels = np.array([1] * len(prompt))  # label all points as forground
        mask, score, logit = predictor.predict(
            point_coords=prompt,
            point_labels=input_labels,
            multimask_output=False,
        )
        masks.append(mask)
    return masks


def get_logits(prompts, predictor):
    """This function returns the models logits of the masks.

    Args:
        prompts (list): List of lists of input points
        predictor (SAMPredictor): Predictor that has already a set image

    Returns:
        np.array: 3D array with the logits from the models output of three given prompts
        (one for each class)
    """

    masks = []
    for label, prompt in enumerate(prompts):
        input_labels = np.array([1] * len(prompt))  # label all points as forground
        mask, score, logit = predictor.predict(
            point_coords=prompt,
            point_labels=input_labels,
            multimask_output=False,
            return_logits=True,
        )  # create mask with highest internal score
        masks.append(mask)
    return masks


def multiclass_prob(binary_logits, hard_labels=False):
    """Get probabilities for multiclass classification.

    Args:
        binary_logits (np.array): Logits generated by SAM for each
        foreground class

    Returns:
        np.array: Multiclass probabilities for different objects and background
    """
    probabilities = []
    for classnr, logit in enumerate(binary_logits):
        # Apply sigmoid function to get probabilities
        probabilities.append(1 / (1 + np.exp(-logit[0])))

    multiclass_prob = np.zeros(
        probabilities[0].shape + (len(binary_logits) + 1,)
    )  # initialize multiclass probabilities
    bin_probs = np.stack(
        probabilities, axis=-1
    )  # combine binary probabilities for foreground classes
    sum_probs = np.sum(bin_probs, axis=-1)
    prob_background = 1 - sum_probs

    mask = sum_probs > 1  # check where to apply our heuristic
    bin_probs[mask] /= sum_probs[mask, None]
    prob_background[mask] = 0

    multiclass_prob[..., 1 : len(binary_logits) + 1] = bin_probs
    multiclass_prob[
        ..., 0
    ] = prob_background  # Use the last channel for prob_background

    if hard_labels == True:
        predicted_labels = np.argmax(multiclass_prob, axis=-1)
        return predicted_labels

    return multiclass_prob
