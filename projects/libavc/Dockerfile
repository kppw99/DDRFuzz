FROM base/ddrfuzz:latest

LABEL maintainer="onsoim <onsoim@gmail.com>"

# Set Environment Variables
ARG home=/targets/libavc
ARG seed=${home}/seed
ARG pro_home=/tool/ddrfuzz/projects/libavc

RUN apt-get update && \
    apt-get install -y wget cmake

RUN git clone https://android.googlesource.com/platform/external/libavc && mkdir libavc/build

WORKDIR /tool/ddrfuzz/
RUN git pull origin main

WORKDIR /tool/libavc/build
RUN CC=afl-clang CXX=afl-clang++ cmake .. && \
    make

ARG TYPE=h264

RUN git clone https://github.com/onsoim/SeeD4Fuzzing && \
    mkdir -p $seed/org && \
    cp -R SeeD4Fuzzing/$TYPE/* $seed/org/ && \
    rm -rf SeeD4Fuzzing

RUN mkdir -p $home/bin && \
    cp avcdec $home/bin/ && \
    cp ../test/decoder/dec.cfg $home/bin/

RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR $home
