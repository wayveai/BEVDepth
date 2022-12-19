# Adapted from https://github.com/TRI-ML/dd3d/blob/main/docker/Dockerfile
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

ENV PYTHON_VERSION=3.8

# -------------------------
# TODO: AZURE credentials
# -------------------------


# -------------------------
# TODO: W&B credentials
# -------------------------
# ARG WANDB_ENTITY
# ENV WANDB_ENTITY=${WANDB_ENTITY}

# ARG WANDB_API_KEY
# ENV WANDB_API_KEY=${WANDB_API_KEY}

# -------------------------
# Install core APT packages (can be simplified)
# -------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    # essential
    build-essential \
    ffmpeg \
    git \
    curl \
    docker.io \
    vim \
    wget \
    unzip \
    htop \
    pkg-config \
    # python
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    python${PYTHON_VERSION}-distutils \
    # set python
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Install BEVDepth requirements
# https://github.com/wayveai/BEVDepth/blob/main/README.md#installation
# -------------------------
WORKDIR /home

# Upgrade pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py


# Other packages listed in BEVDepth requirements.txt
RUN pip install \
    numba \
    numpy \
    nuscenes-devkit \
    opencv-python-headless \
    pandas \
    scikit-image \
    scipy \
    setuptools \
    tensorboardX \ 
    wandb \
    pytorch-lightning==1.5.10 \ 
    torchmetrics==0.10.3

# Install pytorch and torchvision (need to do this last otherwise pytorch-lightning would try to upgrade torch?)
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMx dependencies
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install mmcv-full mmdet mmsegmentation

RUN git clone -b 2.x https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    pip install -e . && \
    python .dev_scripts/check_installation.py

RUN git clone -b 3.x https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && \
    pip install -v -e .

RUN git clone -b 1.x https://github.com/open-mmlab/mmsegmentation.git && \ 
    cd mmsegmentation && \
    pip install -v -e .

RUN git clone -b 1.x https://github.com/open-mmlab/mmclassification.git && \
    cd mmclassification && \
    pip install -e .

RUN git clone -b 1.1 https://github.com/open-mmlab/mmdetection3d.git && \
    cd mmdetection3d && \
    pip install -e . 

# Install BEVDepth
RUN git clone -b francis/dev https://github.com/wayveai/BEVDepth.git && \
    cd BEVDepth && \
    python setup.py develop

# -------------------------
# TODO: Setup nuScenes dataset (download nuscenes dataset from Azure if needed)  
# -------------------------

# Install azcopy 
# Use azcopy to fetch the dataset 

#-----------------------
# Set up final work directory
#-----------------------
WORKDIR /home/BEVDepth
ENV PYTHONPATH "${PYTHONPATH}:/home/BEVDepth"

# For debugging
# download a pretrained checkpoint and run eval script to test if everything works
RUN mkdir bevdepth/exps/nuscenes/ckpt && \
    wget -P bevdepth/exps/nuscenes/ckpt https://github.com/Megvii-BaseDetection/BEVDepth/releases/download/v0.0.2/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da.pth

RUN mkdir data && ln -s /nuscenes data/nuScenes
