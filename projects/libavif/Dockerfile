FROM base/ddrfuzz

LABEL maintainer="onsoim <onsoim@gmail.com>"

# Set Environment Variables
ARG home=/targets/libavif
ARG seed=${home}/seed
ARG pro_home=/tool/ddrfuzz/projects/libavif

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl python3-pip python3-setuptools python3-wheel cmake git unzip && \
    pip3 install meson ninja

RUN git clone --depth 1 https://github.com/AOMediaCodec/libavif.git libavif
WORKDIR /tool/libavif

RUN curl -L https://download.videolan.org/contrib/nasm/nasm-2.14.tar.gz | tar xvz
WORKDIR /tool/libavif/nasm-2.14
RUN ./configure && make -j2 && ln -s `pwd`/nasm /usr/bin/nasm
WORKDIR /tool/libavif
RUN nasm --version

WORKDIR /tool/ddrfuzz/
RUN git pull origin main

# build dav1d
WORKDIR /tool/libavif/ext
RUN bash dav1d.cmd

# build libavif
WORKDIR /tool/libavif
RUN cp -rf ${pro_home}/build_new.sh .
RUN bash build_new.sh
WORKDIR /tool/libavif/build

#RUN mkdir build
#WORKDIR /tool/libavif/build
#RUN CC=afl-clang CXX=afl-clang++ cmake -G Ninja -DAVIF_BUILD_EXAMPLES=on ..
#RUN ninja

ARG TYPE=avif
#     

RUN git clone https://github.com/onsoim/SeeD4Fuzzing && \
    mkdir -p $seed/org && \
    find SeeD4Fuzzing/$TYPE/ -size 1M -exec cp {} $seed/org/ \; && \
    rm -rf SeeD4Fuzzing

RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

RUN mkdir -p $home/bin && \
    cp avif_example_decode_file $home/bin/

WORKDIR $home
