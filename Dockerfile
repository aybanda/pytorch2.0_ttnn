FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        libgl1-mesa-glx \
        git-lfs \
        libsndfile1 \
        docker.io && \
    rm -rf /var/lib/apt/lists/*
