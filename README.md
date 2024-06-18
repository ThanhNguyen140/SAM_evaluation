Name: Phuong Thanh Nguyen and Lisa Krapp

Content
- [LAB PROJECT: EVALUATION OF SEGMENT ANYTHING MODEL IN MEDICAL APPLICATION](#lab-project-evaluation-of-segment-anything-model-in-medical-application)
  - [1. Data](#1-data)
  - [2. Evaluation of SAM](#2-evaluation-of-sam)
  - [3. Modification of SAM](#3-modification-of-sam)
  - [4. Notebooks](#4-notebooks)
  - [5. Functions module](#5-functions-module)

## LAB PROJECT: EVALUATION OF SEGMENT ANYTHING MODEL IN MEDICAL APPLICATION
### 1. Data
- ACDC dataset of Human heart project: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb/folder/637218e573e9f0047faa00fc 
### 2. Evaluation of SAM
- Evaluation of SAM performance by different numbers and combinations of point prompts with different metrics
  1. [pipeline.py](functions/pipeline.py) includes functions for result analysis and experiment pipelines
- Evaluation of SAM by interactive prompts to correct segmentation based on error map
  1. [prompts.py](functions/prompts.py) for interactive prompts based on error map
### 3. Modification of SAM
- Feeding SAM decoder directly by stored embeddings in big batches for efficient processing:
  1. [generate_embeddings.py](functions/generate_embeddings.py) for processing and storing embeddings in batches
  2. [modified_predictor.py](functions/modified_predictor.py) for SAM model to take embeddings instead of images
- Multiple class probabilities function for mutiple classes segmentation
  1. [sam_functions.py](functions/sam_functions.py) includes tools for prompts and multiple class probabilities

### 4. Notebooks
- Examples of using the [functions module](functions) and results analysis
- Including: [testing](testing), [experiments](experiments), [result analysis](result_analysis)

### 5. [Functions](functions) module
- Includes scripts for the whole experiments
  

