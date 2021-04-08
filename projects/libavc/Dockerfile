FROM ddrfuzz/base:latest

LABEL maintainer="onsoim <onsoim@gmail.com>"

# Set Environment Variables
ARG home=/targets/libavc
ARG seed=${home}/seed

RUN apt-get update && \
    apt-get install -y wget cmake

RUN git clone https://android.googlesource.com/platform/external/libavc && mkdir libavc/build

WORKDIR /tool/libavc/build
RUN CC=afl-clang CXX=afl-clang++ cmake .. && \
    make

RUN git clone https://github.com/onsoim/SeeD4Fuzzing && \
    mkdir -p $seed/org && \
    cp -R SeeD4Fuzzing/h264/* $seed/org/ && \
    rm -rf SeeD4Fuzzing

RUN mkdir -p $home/bin && \
    cp avcdec $home/bin/ && \
    cp ../test/decoder/dec.cfg $home/bin/

WORKDIR $home