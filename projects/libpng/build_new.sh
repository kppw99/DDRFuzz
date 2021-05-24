#!/bin/bash

if [[ "$#" -ge "1" && $1 == "cov" ]];
then
    CFLAGS="-fprofile-arcs -ftest-coverage"
    CXXFLAGS="-fprofile-arcs -ftest-coverage"
    LDFLAGS="-fprofile-arcs -ftest-coverage -static"
    LDFLAGS_TEST="-fprofile-arcs -ftest-coverage"

fi

CC=/tool/afl4ddrfuzz/afl-clang
CXX=/tool/afl4ddrfuzz/afl-clang++
nproc=1
OUT=.
SRC=.

make clean
CC=$CC CXX=$CXX CFALGS=$CFLAGS CXXFALGS=$CXXCFLAGS ./configure
#CC=$CC CXX=$CXX ./configure

make -j
