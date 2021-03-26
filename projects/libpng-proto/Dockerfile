
FROM ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

RUN mkdir -p /src && cd /src 
WORKDIR /src 

RUN apt-get install -y make autoconf automake libtool zlib1g-dev binutils cmake ninja-build liblzma-dev libz-dev pkg-config

RUN git clone --depth 1 https://github.com/glennrp/libpng.git
RUN git clone --depth 1 https://github.com/google/libprotobuf-mutator.git
RUN git clone --depth 1 https://github.com/google/fuzzer-test-suite

