
FROM base/ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

# Set Environment Variables
ENV CC=afl-clang
ENV CXX=afl-clang++
ARG home=/targets/libmpeg2
ARG pro_home=/tool/ddrfuzz/projects/libmpeg2


# Prerequisite
WORKDIR /tool/ddrfuzz/
RUN git pull origin main
WORKDIR ${home}/seed/org/
WORKDIR ${home}/seed/copt/
WORKDIR ${home}/seed/topt/
WORKDIR ${home}/bin/
WORKDIR ${home}
RUN git clone https://android.googlesource.com/platform/external/libmpeg2 source
RUN apt-get -y install zip

# Build Target Program
WORKDIR ${home}/source/
RUN mkdir build && cd build && CC=$CC CXX=$CXX cmake .. && make -j


# Copy Seed Corpus and Fuzzing Executables
WORKDIR ${pro_home}
RUN tar -zxvf seed.tar.gz

RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
RUN cp -rf ${home}/source/build/mpeg2dec ${home}/bin/

RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR ${home}


# docker build -t libmpeg2/ddrfuzz:latest . -f Dockerfile
