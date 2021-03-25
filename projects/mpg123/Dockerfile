FROM base/ddrfuzz:latest
MAINTAINER "Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Set Environment Variables
ENV CC=afl-gcc
ENV C++=afl-g++


# Prerequisite
WORKDIR /targets/mpg123/seed/org/
WORKDIR /targets/mpg123/seed/copt/
WORKDIR /targets/mpg123/seed/topt/
WORKDIR /targets/mpg123/bin/
WORKDIR /targets/mpg123/source/
RUN wget https://www.mpg123.de/snapshot
RUN tar -xvf snapshot --directory=/targets/mpg123/source --strip-components=1
RUN rm -rf snapshot


# Build Target Program
WORKDIR /targets/mpg123/source/build/
RUN CFLAGS="-Os -s" ../configure --with-cpu=generic  --disable-id3v2 --disable-lfs-alias --disable-feature-report --with-seektable=0 --disable-16bit --disable-32bit --disable-8bit --disable-messages --disable-feeder --disable-ntom --disable-downsample --disable-icy --disable-shared
RUN make


# Copy Seed Corpus and Fuzzing Executables
RUN cp -rf /tool/ddrfuzz/projects/mpg123/seed/* /targets/mpg123/seed/org/
RUN cp -rf /targets/mpg123/source/build/src/mpg123 /targets/mpg123/bin/

RUN cp -rf /tool/ddrfuzz/projects/mpg123/afl-fuzz.sh /targets/mpg123/
RUN cp -rf /tool/ddrfuzz/projects/mpg123/opt-seed.sh /targets/mpg123/

WORKDIR /targets/mpg123/

# docker build -t mpg123/ddrfuzz:latest . -f Dockerfile
