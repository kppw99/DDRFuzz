
FROM ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

RUN mkdir -p /src && cd /src 
WORKDIR /src 

RUN apt-get install -y make autoconf automake libtool pkg-config zip
RUN git clone --depth 1 https://github.com/libraw/libraw

WORKDIR libraw
