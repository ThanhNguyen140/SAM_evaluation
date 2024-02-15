import torch
import numpy as np


class Prompt:
    def __init__(self, target_class, ground_truth, coordinates, labels):
        self.batched_points = coordinates
        self.batched_labels = labels
        self.batch_size = coordinates.shape[0]
        self.ground_truth = ground_truth

        # transform sam formatted prompts back to image shaped tensors
        self.prompts_tensor_f, self.prompts_tensor_b = self.coordinates_to_masks(
            self.batched_points, self.batched_labels
        )
        # self.prompts_tensor_f contains all foreground points of the prompt
        # self.prompts_tensor_b contains all background points of the prompt

        self.masks = None
        self.error_maps = None

        self.target_class = target_class
        self.class_mask = self.ground_truth == self.target_class

    def coordinates_to_masks(self, coordinates, labels):
        prompts_tensor_f = torch.zeros(
            (
                coordinates.shape[0],
                1,
                self.ground_truth.shape[0],
                self.ground_truth.shape[1],
            ),
            dtype=torch.uint8,
        )
        if coordinates.numel() != 0:  # if the prompt given wasn´t empty
            if (labels == 1).any():  # if not all given labels are background
                # Get the indices where labels are equal to 1 (foreground)
                foreground_indices = torch.nonzero(labels == 1, as_tuple=False)
                foreground_coordinates = coordinates[
                    foreground_indices[:, 0], foreground_indices[:, 1]
                ]
                # Reshape the result to retain the batch dimension
                foreground_coordinates = foreground_coordinates.view(
                    coordinates.shape[0], -1, 2
                )
                # Set the specified points to 1 in the mask tensor

                for i in range(foreground_coordinates.shape[0]):
                    row, col = foreground_coordinates[i, 0]
                    prompts_tensor_f[i, 0, row, col] = 1

        # do everything again for background labeled points
        prompts_tensor_b = torch.zeros(
            (
                coordinates.shape[0],
                1,
                self.ground_truth.shape[0],
                self.ground_truth.shape[1],
            ),
            dtype=torch.uint8,
        )
        # Get the indices where labels are equal to 0 (background)
        background_indices = torch.nonzero(labels == 0, as_tuple=False)
        # check if there is background labeled points
        if background_indices.numel() != 0:
            background_coordinates = coordinates[
                background_indices[:, 0], background_indices[:, 1]
            ]
            # Reshape the result to retain the batch dimension
            background_coordinates = background_coordinates.view(
                coordinates.shape[0], -1, 2
            )

            # Set the specified points to 1 in the mask tensor
            for i in range(background_coordinates.shape[0]):
                row, col = background_coordinates[i, 0]
                prompts_tensor_b[i, 0, row, col] = 1

        return prompts_tensor_f, prompts_tensor_b

    def give_masks(self, masks):
        self.masks = masks
        self.error_maps = self.masks != self.ground_truth.unsqueeze(0)
        return None

    def get_prompts_sam(self):
        return self.batched_points, self.batched_labels

    def add_point_to_prompts(self):
        if self.error_maps == None:
            raise ValueError(
                "self.error_maps is None. Use .give_masks to give existing masks."
            )
        batched_points = []
        batched_labels = []

        for i in range(self.batch_size):
            # Combine class mask and error_maps
            mask_to_sample_from_f = torch.logical_and(
                self.error_maps[i], self.class_mask
            )
            # Exclude already chosen points from sampling
            mask_to_sample_from_f = torch.logical_and(
                ~self.prompts_tensor_f[i].bool(), mask_to_sample_from_f
            )
            # check if there is any true value in mask_to_sample_from_f
            contains_true = torch.any(mask_to_sample_from_f)

            if contains_true:
                # Sample a foreground point for this batch entry
                idx_f = (
                    mask_to_sample_from_f.view(-1)
                    .float()
                    .multinomial(num_samples=1, replacement=False)
                )
                # Update the prompts tensor for this batch entry
                self.prompts_tensor_f[i].view(-1)[idx_f] = 1

            # Sample false positives as background points for batch entries without false negatives
            else:
                # Get the mask for sampling background points for this batch entry
                mask_to_sample_from_b = torch.logical_and(
                    self.error_maps[i], torch.logical_not(self.class_mask)
                )
                mask_to_sample_from_b = torch.logical_and(
                    ~self.prompts_tensor_b[i].bool(), mask_to_sample_from_b
                )
                # Sample a background point for this batch entry
                idx_b = (
                    mask_to_sample_from_b.view(-1)
                    .float()
                    .multinomial(num_samples=1, replacement=False)
                )
                # Update the prompts tensor for this batch entry
                self.prompts_tensor_b[i].view(-1)[idx_b] = 1

            # update the prompts in the SAM format
            foreground_points = torch.nonzero(self.prompts_tensor_f[i])[:, 1:]
            foreground_points[:, [0, 1]] = foreground_points[:, [1, 0]]  # swap axes
            f_labels = torch.tensor([1] * len(foreground_points))

            background_points = torch.nonzero(self.prompts_tensor_b[i])[:, 1:]
            background_points[:, [0, 1]] = background_points[:, [1, 0]]  # swap axes
            b_labels = torch.tensor([0] * len(background_points))

            points = torch.cat([foreground_points, background_points], dim=0)
            labels = torch.cat([f_labels, b_labels], dim=0)
            batched_points.append(points)
            batched_labels.append(labels)

        self.batched_points = torch.stack(batched_points)
        self.batched_labels = torch.stack(batched_labels)

        return None
