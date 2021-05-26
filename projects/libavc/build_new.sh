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

if [ -d build ];
then
	rm -rf ./build
	mkdir build
else
	mkdir build
fi


cd build
CC=$CC CXX=$CXX cmake ..
make
echo $CC


