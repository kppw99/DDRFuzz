
FROM ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

RUN mkdir -p /src && cd /src 
WORKDIR /src 

RUN apt-get install -y make autoconf automake libtool zlib1g-dev

RUN git clone --depth 1 https://github.com/madler/zlib.git
RUN git clone --depth 1 https://github.com/glennrp/libpng.git


WORKDIR libpng
