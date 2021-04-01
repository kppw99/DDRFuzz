
FROM base/ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

# Set Environment Variables
ENV CC=afl-gcc
ENV CXX=afl-g++
ARG home=/targets/libspng
ARG pro_home=/tool/ddrfuzz/projects/libspng


# Prerequisite
WORKDIR /tool/ddrfuzz/
RUN git pull origin main
WORKDIR ${home}/seed/org/
WORKDIR ${home}/seed/copt/
WORKDIR ${home}/seed/topt/
WORKDIR ${home}/bin/
WORKDIR ${home}
RUN apt-get install -y wget tar cmake
RUN git clone --depth 1 https://github.com/randy408/libspng.git source
RUN git clone --depth 1 https://github.com/google/fuzzer-test-suite
RUN wget -O zlib.tar.gz http://www.zlib.net/zlib-1.2.11.tar.gz
RUN tar xzvf zlib.tar.gz --directory .
RUN mv zlib-1.2.11 zlib


# Build Target Program
WORKDIR ${home}/zlib/
RUN mkdir build && cd build && cmake .. && make -j && make install
WORKDIR ${home}/source/
RUN mkdir build && cd build && CC=$CC CXX=$CXX cmake .. && make


# Copy Seed Corpus and Fuzzing Executables
#RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
#RUN cp -rf ${home}/source/.libs/pngtest ${home}/bin/

#RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
#RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR ${home}


# docker build -t libmpeg2/ddrfuzz:latest . -f Dockerfile
