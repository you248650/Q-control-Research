FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 # Macなら ubuntu20.04 etc.

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/RL4Quantum/.setup/

COPY ./Docker/requirements_dependence.txt .
COPY ./Docker/requirements.txt .

WORKDIR /home/RL4Quantum/

COPY . .

RUN apt-get update && \
    apt-get install -y eog firefox git python3.8 python3-pip python3-tk vim && \
    pip install -r /home/RL4Quantum/.setup/requirements_dependence.txt && \
    pip install -r /home/RL4Quantum/.setup/requirements.txt && \
    echo "alias python='/usr/bin/python3'" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc" && \
    /bin/bash -c "rm -rf Docker/" && \
    /bin/bash -c "rm -rf .devcontainer/" && \
    /bin/bash -c "rm -rf .setup"