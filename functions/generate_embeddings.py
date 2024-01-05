# this file should be executed in the folder where the SAM model is stored
# the path to the preprocessed images file should be given

import os
import numpy as np
import nibabel as nib
import sys
import cv2
import gzip
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np
from segment_anything.modeling import Sam
from typing import Optional, Tuple

if __name__ == "__main__":
    # load SAM Model
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    file_path = sys.argv[1]
    with gzip.open(file_path, "rb") as f:
        # Load the NumPy array from the file
        images = np.load(f)

    embeddings = []
    # loop through all images in preprocess
    for image in tqdm(images):
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        sam_img = cv2.normalize(
            img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        sam_img = sam_img.astype(np.uint8)
        predictor.set_image(sam_img)
        embedding = predictor.get_image_embedding()
        embeddings.append(embedding)

    # concat all embeddings to a big tensor
    combined_embeddings = torch.cat(embeddings, dim=0)
    torch.save(combined_embeddings, "embeddings.pt")
