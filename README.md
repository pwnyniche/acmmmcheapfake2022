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
- GPU with minimum 10GB memory .
We recommend using Docker with GPU flag on.

## Install with Docker
### Build the image
    docker build -t acmmmcheapfakes:submission .
### or Pull from dockerhub
    docker pull tqtnk2000/acmmmcheapfakes:submission
### Run the image
    docker run -v  <path to folder containing the hidden test split file test.json>:/acmmmcheapfakes/  --gpus all acmmmcheapfakes:submission > <output file>

## The folder containing the test split file should look like this (for both tasks):
Please note that a JSON file containing annotation for Task 1 and 2 **must** be named `test_data.json` and `task_2.json` respectively.

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

We also include the sample structure in `data` folder.


