FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        libgl1-mesa-glx \
        git-lfs \
        libsndfile1 \
        docker.io \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY . /app
WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
