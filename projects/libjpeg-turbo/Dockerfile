FROM base/ddrfuzz:latest
LABEL maintainer="onsoim <onsoim@gmail.com>"

RUN apt-get update && \
    apt-get install -y make autoconf automake libtool nasm curl cmake

RUN git clone --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo && \
    mkdir libjpeg-turbo/build

WORKDIR /tool/libjpeg-turbo/build/
RUN CC=afl-clang CXX=afl-clang++ cmake ../ && make "-j$(nproc)"

RUN mkdir input
WORKDIR /tool/libjpeg-turbo/build/input
RUN curl https://lcamtuf.coredump.cx/afl/demo/afl_testcases.tgz | tar -xz

WORKDIR /tool/libjpeg-turbo/build

# afl-fuzz -m 1024 -i input/bmp/full/images -o output ./cjpeg @@