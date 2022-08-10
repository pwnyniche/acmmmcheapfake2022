# A Textual-Visual-Entailment-based Unsupervised Algorithm for Cheapfake Detection

## Requirements
- python 3.7
- cuda 11.1
- pytorch 1.9.0
- torchvision 0.10.0 + torchtext 0.4.0
- spacy
- COSMOS (included in the code, just install the requirements.txt)
- OFA (included in the code, just install the requirements.txt)
- detectron2 (included in the code, just install the requirements.txt)
- GPU with minimum 10GB memory

_Note: We recommend using Docker with GPU for easy installation._

## Install with Docker
### Build the image
    docker build -t acmmmcheapfakes:submission .
### or Pull from dockerhub
    docker pull tqtnk2000/acmmmcheapfakes:submission
### Run the image
    docker run -v  <path to folder containing the hidden test split file test.json>:/acmmmcheapfakes/  --gpus all acmmmcheapfakes:submission > <output file>

## Input / Output:
### Input:
The input includes an annotation file and an image folder from the COSMOS dataset. The annotation file should include the captions and the relative path to the raw image.

### Output:
The output is a dataframe with predicted labels (0 for NOOC, 1 for OOC) and a confusion matrix. The dataframe also includes a field to indicate which method is used to predict the label. Please modify the code in "acmmm.py" to save this dataframe for further inspection.

## How the code run:
The main code is in the file "acmmm.py". Each of the components is written separatedly after the command **print("Runnning _method_")** 

## Dataset:
### The COSMOS dataset:
The COSMOS dataset is not public. Please visit https://detecting-cheapfakes.github.io/ or fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSf7rZ1-UX419nXqCp2NldekqVNJcS2W9A3jL7MTKhom41p0eg/viewform) to get access.
### Dataset folder structure:
The folder containing the test split file should look like this (for both tasks):

    data
    ├── images_task_2            
    │   ├── 2.jpg                
    │   ├── 20.jpg        
    │   ├── 58.jpg      
    │   └── ...      
    ├── public_test_mmsys          
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   ├── 2.jpg  
    │   └── ...          
    ├── task_2.json 
    └── test_data.json

Please note that a JSON file containing annotation for Task 1 and 2 **must** be named `test_data.json` and `task_2.json` respectively.



