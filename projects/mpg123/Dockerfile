FROM base/ddrfuzz:latest
MAINTAINER "Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Set Environment Variables
ENV CC=afl-gcc
ENV CXX=afl-g++
ARG home=/targets/mpg123
ARG pro_home=/tool/ddrfuzz/projects/mpg123

# Prerequisite
WORKDIR ${home}/seed/org/
WORKDIR ${home}/seed/copt/
WORKDIR ${home}/seed/topt/
WORKDIR ${home}/bin/
WORKDIR ${home}/source/
RUN wget https://www.mpg123.de/snapshot
RUN tar -xvf snapshot --directory=${home}/source --strip-components=1
RUN rm -rf snapshot


# Build Target Program
WORKDIR ${home}/source/build/
RUN CFLAGS="-Os -s" ../configure --with-cpu=generic  --disable-id3v2 --disable-lfs-alias --disable-feature-report --with-seektable=0 --disable-16bit --disable-32bit --disable-8bit --disable-messages --disable-feeder --disable-ntom --disable-downsample --disable-icy --disable-shared
RUN make


# Copy Seed Corpus and Fuzzing Executables
WORKDIR ${pro_home}
RUN tar -zxvf seed.tar.gz

RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
RUN cp -rf ${home}/source/build/src/mpg123 ${home}/bin/

RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR ${home}

# docker build -t mpg123/ddrfuzz:latest . -f Dockerfile
