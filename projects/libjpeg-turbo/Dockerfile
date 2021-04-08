FROM ddrfuzz/base:latest

LABEL maintainer="onsoim <onsoim@gmail.com>"

# Set Environment Variables
ARG home=/targets/libjpeg-turbo
ARG seed=${home}/seed

RUN apt-get update && \
    apt-get install -y make autoconf automake libtool nasm curl cmake

RUN git clone --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo && \
    mkdir libjpeg-turbo/build

WORKDIR /tool/libjpeg-turbo/build/
RUN CC=afl-clang CXX=afl-clang++ cmake ../ && make "-j$(nproc)"

ARG TYPE=bmp

RUN git clone https://github.com/onsoim/SeeD4Fuzzing && \
    mkdir -p $seed/org && \
    cp -R SeeD4Fuzzing/$TYPE/* $seed/org/ && \
    rm -rf SeeD4Fuzzing

RUN mkdir -p $home/bin && \
    cp cjpeg $home/bin/

WORKDIR $home