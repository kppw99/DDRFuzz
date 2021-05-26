#!/bin/bash

if [[ "$#" -ge "1" && $1 == "cov" ]];
then
	CC="/tool/afl4ddrfuzz/afl-gcc -fprofile-arcs -ftest-coverage"
	CXX="/tool/afl4ddrfuzz/afl-g++ -fprofile-arcs -ftest-coverage"

else
	CC="/tool/afl4ddrfuzz/afl-gcc"
	CXX="/tool/afl4ddrfuzz/afl-g++"
fi

nproc=1
OUT=.
SRC=.

make clean
./autogen.sh
CC=$CC CXX=$CXX ./configure --disable-shared
make -j8
