#!/bin/bash

if [[ "$#" -ge "1" && $1 == "cov" ]];
then
	CC="/tool/afl4ddrfuzz/afl-clang -fprofile-arcs -ftest-coverage"
	CXX="/tool/afl4ddrfuzz/afl-clang++ -fprofile-arcs -ftest-coverage"

else
	CC="/tool/afl4ddrfuzz/afl-clang"
	CXX="/tool/afl4ddrfuzz/afl-clang++"
fi

OUT=.
SRC=.

make clean
CC=$CC CXX=$CXX CFLAGS="-Os -s" ../configure --with-cpu=generic  --disable-id3v2 --disable-lfs-alias --disable-feature-report --with-seektable=0 --disable-16bit --disable-32bit --disable-8bit --disable-messages --disable-feeder --disable-ntom --disable-downsample --disable-icy --disable-shared
make
