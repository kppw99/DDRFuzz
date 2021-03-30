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

RUN apt-get install -y zip && \
    wget https://storage.googleapis.com/android_media/external/libavc/fuzzer/avc_dec_fuzzer_seed_corpus.zip && \
    unzip avc_dec_fuzzer_seed_corpus.zip -d input

## need to be run with permission
# echo core >/proc/sys/kernel/core_pattern
# afl-fuzz -m 1024 -i input/h264/ -o output ./avcdec --input @@