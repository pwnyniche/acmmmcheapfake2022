# A Textual-Visual-Entailment-based Unsupervised Algorithm for Cheapfake Detection

## Build the image
    docker build -t acmmmcheapfakes:submission .

## Run the image
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



* Tải bộ dữ liệu test:
wget http://kaldir.vc.in.tum.de/images_test_acm.zip -q --show-progress
* Sau khi unzip, dữ liệu bao gồm 1 file annotation và 1 folder chứa ảnh.

Để chạy code, người dùng có thể tải Image từ Dockerhub hoặc build Image bằng Dockerfile có trong Source:
Cách 1: 
B1: Để tải Docker Image từ Dockerhub và chạy trực tiếp, vui lòng gõ lệnh sau:
``` 
docker pull tqtnk2000/acmmmcheapfakes:submission
```
B2: Mount folder chứa dữ liệu đã tải về vào docker và bật GPU để chạy code:
```
docker run -v <path to folder containing the hidden test split file test.json>:/acmmmcheapfakes --gpus all acmmmcheapfakes:submission > <output file>
```
Cách 2:
B1: Chuyển đến thư mục chứ code và build Docker Image từ source code
```
docker build -t acmmmcheapfakes:submission .
```
B2: Sau khi build, mount folder chứa dữ liệu đã tải về vào docker, bật GPU và chạy image:
```
docker run -v <path to folder containing the hidden test split file test.json>:/acmmmcheapfakes/ --gpus all acmmmcheapfakes:submission > <output file>
```

*** Lưu ý: File JSON phải được đặt tên là `test_data.json` và folder chứa ảnh là `public_test_mmsys`:
    data
    ├── public_test_mmsys          
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   ├── 2.jpg  
    │   └── ...          
    └── test_data.json

