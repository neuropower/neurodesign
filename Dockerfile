FROM debian:8
MAINTAINER Joke Durnez

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

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
