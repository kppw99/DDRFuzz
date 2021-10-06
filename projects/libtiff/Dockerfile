FROM base/ddrfuzz:latest
MAINTAINER "Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Set Environment Variables
ENV CC=afl-gcc
ENV CXX=afl-g++
ARG home=/targets/libtiff
ARG pro_home=/tool/ddrfuzz/projects/libtiff


# Prerequisite
WORKDIR /tool/ddrfuzz/
RUN git pull origin main
WORKDIR ${home}/seed/org/
WORKDIR ${home}/seed/copt/
WORKDIR ${home}/seed/topt/
WORKDIR ${home}/bin/
WORKDIR ${home}
RUN git clone --depth 1 https://github.com/vadz/libtiff.git source


# Build Target Program
WORKDIR ${home}/source/
RUN cp -rf ${pro_home}/build_new.sh ${home}/source/
RUN apt-get -y install m4 autoconf libtool zlib1g-dev liblzma-dev
RUN bash build_new.sh
#RUN ./autogen.sh
#RUN ./configure --disable-shared
#RUN make -j8


# Copy Seed Corpus and Fuzzing Executables
WORKDIR ${pro_home}
RUN tar -zxvf seed.tar.gz

RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
RUN cp -rf ${pro_home}/valuable_seed.zip ${home}
RUN cp -rf ${home}/source/tools/bmp2tiff ${home}/bin/

RUN cp -rf ${pro_home}/afl-ddrfuzz.sh ${home}
RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR ${home}

# docker build -t libtiff/ddrfuzz:latest . -f Dockerfile
