
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
RUN git clone --depth 1 https://github.com/glennrp/libpng.git
RUN apt-get install -y make autoconf automake libtool zlib1g-dev

# Build Target Program
WORKDIR ${home}/source/
RUN ./configure CC=$CC CXX=$CXX && make -j


# Copy Seed Corpus and Fuzzing Executables
#RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
#RUN cp -rf ${home}/source/build/mpeg2dec ${home}/bin/

#RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
#RUN cp -rf ${pro_home}/opt-seed.sh ${home}

#WORKDIR ${home}


# docker build -t libmpeg2/ddrfuzz:latest . -f Dockerfile
