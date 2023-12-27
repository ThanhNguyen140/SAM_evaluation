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
    """Subclass of SamPredictor class. This class allows the generation of masks with the same syntax as the parent class for predict function
    using image embeddings
    """
    def __init__(self,image_embedding, input_size, original_size, sam_model = sam):
        """Input for subclass of SamPredictor

        Args:
            image_embedding (NDArray): An image embedding generated by the parent class by set_image
            input_size (tuple): a tuple containing size of image before generating embeddings. This is not
            the orginal size of image because the image is resized according to sam model before generating embeddings
            original_size (tuple): original size of the image in 2D (H x W)
            sam_model (sam model, optional): Defaults to sam.
        """
        super().__init__(sam_model)
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.features = image_embedding
        self.is_image_set = True
        self.input_size = input_size
        self.original_size = original_size

    