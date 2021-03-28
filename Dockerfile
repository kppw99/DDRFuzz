FROM ubuntu:18.04
LABEL maintainer="Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Install Prereqisite Utils
RUN apt-get update --fix-missing
RUN apt-get install -y apt-utils git wget make vim cmake gnupg gcc-multilib g++-multilib apt-transport-https tar


# Install LLVM Development Package
RUN echo "#LLVM Repository" >> /etc/apt/sources.list && \
    echo "deb http://apt.llvm.org/stretch/ llvm-toolchain-stretch-6.0 main" >> /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/stretch/ llvm-toolchain-stretch-6.0 main" >> /etc/apt/sources.list && \
    wget -O ./key.gpg https://apt.llvm.org/llvm-snapshot.gpg.key --no-check-certificate && \
    apt-key add < ./key.gpg && \
    rm ./key.gpg && \
    apt-get -y update && \
    apt-get -y install clang-6.0 clang-6.0-dev llvm-6.0 llvm-6.0-dev && \
    ln -s /usr/bin/clang-6.0 /usr/bin/clang && \
    ln -s /usr/bin/clang++-6.0 /usr/bin/clang++


# Install AFL (2.52.b)
RUN mkdir -p tool && cd /tool && \
    wget http://lcamtuf.coredump.cx/afl/releases/afl-2.52b.tgz && \
    tar zxvf afl-2.52b.tgz && \
    rm afl-2.52b.tgz && \
    cd /tool/afl-2.52b && \
    make


# Download ddrfuzz
WORKDIR /tool/
RUN git clone https://github.com/kppw99/ddrfuzz.git


# Set Environment Variables
ENV PATH=$PATH:/tool/afl-2.52b
ENV AFL_PATH=/tool/afl-2.52b

RUN echo "alias q='cd ..'" >> ~/.bashrc
RUN echo "alias qq='cd ../..'" >> ~/.bashrc
RUN echo "alias qqq='cd ../../..'" >> ~/.bashrc

# docker build -t base/ddrfuzz:latest . -f Dockerfile
