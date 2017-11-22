FROM ubuntu:14.04
MAINTAINER Joke Durnez

RUN ln -snf /bin/bash /bin/sh
ENV SHELL /bin/bash

# Update packages and install the minimal set of tools
RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install -y build-essential && \
    apt-get install -y curl git xvfb bzip2 apt-utils && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y libfreetype6-dev libxft-dev
ENV LANG C.UTF-8

# Install conda
RUN curl -O https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh && \
    yes "yes" | bash Anaconda3-4.2.0-Linux-x86_64.sh && \
    sudo -s source ~/.bashrc

RUN conda install nose && \
    conda install numpy && \
    conda install scipy && \
    conda install pandas && \
    conda install matplotlib && \
    conda install xvfbwrapper

RUN pip install neurodesign > 0.1.4
RUN pip install sklearn
RUN pip install pdfrw
RUN pip install reportlab
RUN pip install progressbar

# Clear apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
