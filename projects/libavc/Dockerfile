FROM base/ddrfuzz:latest

LABEL maintainer="onsoim <onsoim@gmail.com>"

# Set Environment Variables
ENV CC=afl-gcc
ENV C++=afl-g++

RUN apt-get update && \
    apt-get install -y wget cmake unzip

RUN git clone https://android.googlesource.com/platform/external/libavc && mkdir libavc/build

WORKDIR /tool/libavc/build
RUN CC=afl-clang CXX=afl-clang++ cmake .. && \
    make

RUN git clone https://github.com/onsoim/SeeD4Fuzzing && \
    cp -R SeeD4Fuzzing/h264/ input/ && \
    rm -rf SeeD4Fuzzing

## need to be run with permission
# echo core >/proc/sys/kernel/core_pattern
# afl-fuzz -m 1024 -i input/h264/ -o output ./avcdec --input @@