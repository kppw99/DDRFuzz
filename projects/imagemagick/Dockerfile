FROM base/ddrfuzz:latest
MAINTAINER "Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Set Environment Variables
ENV CC=afl-gcc
ENV CXX=afl-g++
ARG home=/targets/imagemagick
ARG pro_home=/tool/ddrfuzz/projects/imagemagick


# Prerequisite
WORKDIR /tool/ddrfuzz/
RUN git pull origin main
WORKDIR ${home}/seed/org/
WORKDIR ${home}/seed/copt/
WORKDIR ${home}/seed/topt/
WORKDIR ${home}/bin/
WORKDIR ${home}
RUN git clone --depth 1 https://github.com/imagemagick/imagemagick source


# Build Target Program
WORKDIR ${home}/source/
RUN ./configure --with-bzlib=no --with-djvu=no --with-dps=no --with-fftw=no --with-fpx=no --with-fontconfig=no --with-freetype=no --with-gvc=no --with-jbig=no --with-jpeg=no --with-lcms=no --with-lqr=no --with-lzma=no --with-openexr=no --with-openjp2=no --with-pango=no --with-png=no --with-tiff=no --with-raqm=no --with-webp=no --with-wmf=no --with-x=no --with-xml=no --with-zlib=no --enable-hdri=no --enable-shared=no
RUN make


# Copy Seed Corpus and Fuzzing Executables
WORKDIR ${pro_home}
RUN tar -zxvf seed.tar.gz

RUN cp -rf ${pro_home}/seed/* ${home}/seed/org/
RUN cp -rf ${home}/source/utilities/magick ${home}/bin/

RUN cp -rf ${pro_home}/afl-fuzz.sh ${home}
RUN cp -rf ${pro_home}/opt-seed.sh ${home}

WORKDIR ${home}

# docker build -t imagemagick/ddrfuzz:latest . -f Dockerfile
