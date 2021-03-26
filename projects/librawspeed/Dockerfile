
FROM ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

RUN mkdir -p /src && cd /src 
WORKDIR /src 

RUN apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main' && \
    apt-get update && apt-get install -y cmake make
RUN git clone --depth 1 https://github.com/darktable-org/rawspeed.git librawspeed
WORKDIR librawspeed

