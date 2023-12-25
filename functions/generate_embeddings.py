import os
import numpy as np
import nibabel as nib
import sys
import glob
import gzip
from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np
from segment_anything.modeling import Sam
from typing import Optional, Tuple
from segment_anything.utils.transforms import ResizeLongestSide

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

# loop through all images in preprocess
z_stack_embedding = []
# loop through z-stacks
predictor = SamPredictor(sam)
predictor.set_image(image)
embedding = predictor.get_image_embedding()
z_stack_embedding.append(embedding)
# save z stack embedding in the folder for each patient with _emb at the end of the filename




class modifiedPredictor((SamPredictor)):
    def __init__(self,sam_model: Sam,image_embedding, input_size, original_size)
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.features = image_embedding
        self.is_image_set = True
        self.input_size = input_size
        self.original_size = original_size

    