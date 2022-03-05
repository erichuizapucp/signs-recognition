FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get install python3.9 -y
RUN apt-get install python3.9-distutils -y
RUN apt-get install python3.9-dev -y
RUN apt-get install python3-pip -y
RUN python3.9 -m pip install --upgrade setuptools
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install --upgrade distlib

COPY src src
COPY requirements.txt .
COPY *.yaml .
COPY train-swav.sh .

RUN python3.9 -m pip install -r requirements.txt
