#!/bin/bash

if [[ "$#" -ge "1" && $1 == "cov" ]];
then
	CC="/tool/afl4ddrfuzz/afl-clang -fprofile-arcs -ftest-coverage"
	CXX="/tool/afl4ddrfuzz/afl-clang++ -fprofile-arcs -ftest-coverage"

else
	CC="/tool/afl4ddrfuzz/afl-clang"
	CXX="/tool/afl4ddrfuzz/afl-clang++"
fi

nproc=1
OUT=.
SRC=.

make clean
CC=$CC CXX=$CXX ./configure

make -j
