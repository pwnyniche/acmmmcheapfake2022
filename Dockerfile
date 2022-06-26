FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Setup Environment Variables
ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Asia/Vietnam"

ENV ANNOTATION_DATA_DIR="/acmmmcheapfakes/" \
    IMAGE_DATA_DIR="/acmmmcheapfakes/" \
    BASE_DIR="/acmmmcheapfakes-src/"

WORKDIR /acmmmcheapfakes-src/

# Copy Dependencies
COPY . ./

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common libcurl4-openssl-dev libssl-dev wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3-dev python3-pip python3-opencv \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Prepare Python Dependencies
RUN python3 -m pip install --upgrade pip==21.2.2 && \
    pip3 install cython numpy setuptools wheel

RUN cd OFA && pip3 install --use-feature=in-tree-build ./fairseq

# Download OFA checkpoint
RUN mkdir OFA/checkpoints
# RUN cd OFA/checkpoints && wget --no-verbose --show-progress --progress=bar:force:noscroll https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/snli_ve_large_best.pt 
RUN pip3 install gdown
RUN cd OFA/checkpoints && gdown 1iFGMW1fEAuQfI02SeV95MGDMu9LGQz-Y

# COSMOS
RUN pip3 install -r COSMOS/requirements.txt

# # Download spaCy
RUN python3 -m spacy download en && \
    python3 -m spacy download en_core_web_sm

# install Pytorch with CUDA
RUN pip3 install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchtext==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install numpy --upgrade

# OFA
RUN cd OFA && pip3 install -r requirements.txt

RUN pip3 install -e detectron2
RUN pip3 install omegaconf==2.0.06
RUN pip3 install hydra-core==1.0.7

# Start the code
ENTRYPOINT []
CMD [ "python3", "acmmm.py" ]