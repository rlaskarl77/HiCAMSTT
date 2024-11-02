FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    git \
    openssh-server && \
    rm -rf /var/lib/apt/lists/*
# set up conda
ENV PATH=/opt/conda/bin:$PATH
RUN conda update -n base -c defaults conda
RUN conda init bash \
    && . ~/.bashrc
# install conda-pack
RUN set -ex && \
    conda config --set always_yes yes --set changeps1 no && \
    conda info -a && \
    conda config --add channels conda-forge && \
    conda install --quiet --freeze-installed -c main conda-pack
# install jupyter
RUN pip install jupyter
# expose jupyter port
RUN ln -s /opt/conda/bin/jupyter /usr/local/bin/jupyter
# install jupyterlab
RUN pip install jupyterlab
# install requirements
RUN pip install \
    packaging \
    triton \
    timm \
    pytest \
    chardet \
    yacs \
    termcolor \
    submitit \
    tensorboardX \
    fvcore \
    seaborn \
    opencv-python \
    tensorboard
# install mamba
ENV CUDA_HOME=/usr/local/cuda
RUN CUDA_HOME=/usr/local/cuda pip install mamba-ssm==2.0.4

# install mmcv (for detection and semgentation)
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1

RUN pip install opencv-python-headless ftfy regex

# Install system dependencies for opencv-python
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmcv
ARG MMCV="2.1.0"
RUN pip install -U openmim

RUN CUDA_HOME=/usr/local/cuda && mim install mmengine==0.10.1
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6 9.0+PTX"
RUN CUDA_HOME=/usr/local/cuda && mim install mmcv==2.1.0

# Verify the installation
RUN python -c 'import mmcv;print(mmcv.__version__)'
RUN python -c 'import mmengine;print(mmengine.__version__)'

RUN CUDA_HOME=/usr/local/cuda && pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
# install FFCV
RUN conda install cupy pkg-config libjpeg-turbo opencv numba -c pytorch -c conda-forge \
    && pip install ffcv
# install lmdb & install pillow-simd
RUN pip install lmdb
RUN pip uninstall -y pillow
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
# install vmtouch
RUN git clone https://github.com/hoytech/vmtouch.git && \
    cd vmtouch && \
    make && \
    make install
# start
ENTRYPOINT [ "/bin/bash" ]
