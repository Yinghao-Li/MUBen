FROM pytorch/pytorch:latest

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        tmux \
        screen \
        unzip

WORKDIR /proj
RUN git clone https://github.com/Yinghao-Li/MUBen.git --single-branch

WORKDIR /proj/MUBen
RUN pip install -r requirements.txt

