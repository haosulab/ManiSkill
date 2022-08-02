FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    tmux \
    wget \
    unzip \
    htop \
    rsync \
    ca-certificates \
    bash-completion \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python=3.8 && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

COPY environment.yml /root
RUN conda env update -n base -f /root/environment.yml \
    && pip install gym==0.18.3 --no-deps \
    && conda clean -ya && pip cache purge
