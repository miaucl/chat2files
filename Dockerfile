FROM ubuntu:22.04

ARG PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y software-properties-common curl
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    swig \
    build-essential \
    libopenblas-dev \
    libomp-dev
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.11 get-pip.py && rm get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#COPY . .
