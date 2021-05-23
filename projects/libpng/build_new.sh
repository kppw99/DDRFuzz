#!/bin/bash

if [[ "$#" -ge "1" && $1 == "cov" ]];
then
    CFLAGS="-fprofile-arcs -ftest-coverage"
    CXXFLAGS="-fprofile-arcs -ftest-coverage"
    LDFLAGS="-fprofile-arcs -ftest-coverage -static"
    LDFLAGS_TEST="-fprofile-arcs -ftest-coverage"
else
    LDFLAGS="-static"
fi

CC=/tool/afl4ddrfuzz/afl-clang
CXX=/tool/afl4ddrfuzz/afl-clang++
nproc=1
OUT=.
SRC=.

make clean
CC=$CC CXX=$CXX CFALGS=$CFLAGS CXXFALGS=$CXXCFLAGS LDFLAGS=$LDFLAGS ./configure

if [[ "$#" -ge "1" && ( $1 == "cov" ) ]];
then
	make -j
	make check
else
	AFL_USE_ASAN=1 make -j
	ASAN_OPTIONS=detect_leaks=0 AFL_USE_ASAN=1 make check
fi

