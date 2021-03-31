FROM ddrfuzz/base
LABEL maintainer="onsoim <onsoim@gmail.com>"

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

# build dav1d
WORKDIR /tool/libavif/ext
RUN bash dav1d.cmd

# build libavif
RUN mkdir build
WORKDIR /tool/libavif/build
RUN CC=afl-clang CXX=afl-clang++ cmake -G Ninja -DAVIF_BUILD_EXAMPLES=on ..
RUN ninja

ADD ossfuzz/avif_decode_seed_corpus.zip .
RUN unzip avif_decode_seed_corpus.zip -d input

# afl-fuzz -m 1024 -i input/ -o output/ ./avif_example_decode_file @@