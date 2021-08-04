FROM base/ddrfuzz:latest
MAINTAINER "Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Set Environment Variables
ENV CC=afl-gcc
ENV CXX=afl-g++
ARG home=/targets/ffmpeg
ARG pro_home=/tool/ddrfuzz/projects/ffmpeg


# Prerequisite
WORKDIR /tool/ddrfuzz/
RUN git pull origin main
WORKDIR ${home}/seed/org/
WORKDIR ${home}/seed/copt/
WORKDIR ${home}/seed/topt/
WORKDIR ${home}/bin/
WORKDIR ${home}
RUN git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git source


# Build Target Program
WORKDIR ${home}/source/
RUN apt-get -y install yasm libass-dev libmp3lame-dev libopus-dev libvorbis-dev libx264-dev libx265-dev libnuma-dev
RUN cp -rf ${pro_home}/build_new.sh .
RUN bash build_new.sh


# Copy Seed Corpus and Fuzzing Executables
WORKDIR ${pro_home}
RUN tar -zxvf seed.tar.gz

RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
RUN cp -rf ${home}/source/ffmpeg ${home}/bin/

RUN cp -rf ${pro_home}/afl-ddrfuzz.sh ${home}
RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR ${home}

# docker build -t ffmpeg/ddrfuzz:latest . -f Dockerfile
