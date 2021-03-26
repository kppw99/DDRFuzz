
FROM ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

RUN mkdir -p /src && cd /src 
WORKDIR /src 

RUN apt-get install -y wget tar cmake


RUN git clone --depth 1 https://github.com/randy408/libspng.git
RUN git clone --depth 1 https://github.com/google/fuzzer-test-suite

RUN wget -O zlib.tar.gz http://www.zlib.net/zlib-1.2.11.tar.gz
RUN tar xzvf zlib.tar.gz --directory .
RUN mv zlib-1.2.11 zlib

WORKDIR libspng
