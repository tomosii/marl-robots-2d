FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

RUN apt-key del 7fa2af80 && \
    apt-key del F60F4B3D7FA2AF80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Install git
RUN apt-get install -y git

# Install pytorch
RUN pip3 install torch torchvision torchaudio 

# Install other dependencies
RUN pip3 install numpy pyyaml pygame wandb sacred

WORKDIR /workspace




