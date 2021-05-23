
FROM base/ddrfuzz:latest
MAINTAINER "Dongyoung Kim <ehddud758@gmail.com>"

# Set Environment Variables
ENV CC=afl-clang
ENV CXX=afl-clang++
ARG home=/targets/libpng
ARG pro_home=/tool/ddrfuzz/projects/libpng


# Prerequisite
WORKDIR /tool/ddrfuzz/
RUN git pull origin main
WORKDIR ${home}/seed/org/
WORKDIR ${home}/seed/copt/
WORKDIR ${home}/seed/topt/
WORKDIR ${home}/bin/
WORKDIR ${home}
RUN git clone --depth 1 https://github.com/madler/zlib.git
RUN git clone --depth 1 https://github.com/glennrp/libpng.git source
RUN apt-get install -y make autoconf automake libtool zlib1g-dev

# Build Target Program
WORKDIR ${home}/source/
RUN cp -rf ${pro_home}/build_new.sh ${home}/source/
ENV LIBRARY_PATH=/targets/libpng/source/.libs/libpng16.so.16
RUN bash build_new.sh
#RUN ./configure CC=$CC CXX=$CXX && make -j


# Copy Seed Corpus and Fuzzing Executables
WORKDIR ${pro_home}
RUN tar -zxvf seed.tar.gz

RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
RUN cp -rf ${home}/source/.libs/pngtest ${home}/bin/

RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR ${home}


# docker build -t libpng/ddrfuzz:latest . -f Dockerfile
