#!/bin/bash

CC="/tool/afl4ddrfuzz/afl-clang"
CXX="/tool/afl4ddrfuzz/afl-clang++"

make distclean
make clean

if [[ "$#" -ge "1" && $1 == "cov" ]];
then
	./configure --enable-cross-compile --toolchain=gcov --prefix="$HOME/ffmpeg_build" --pkg-config="pkg-config --static" --extra-cflags="-I$HOME/ffmpeg_build/include" --extra-ldflags="-L$HOME/ffmpeg_build/lib" --extra-libs="-lpthread -lm" --bindir="$HOME/bin" --enable-gpl --enable-libass --enable-libfreetype --enable-libmp3lame --enable-libopus --enable-libvorbis --enable-libx264 --enable-libx265 --enable-nonfree  --extra-cflags="-I$HOME/ffmpeg_build/include -O1 -fno-omit-frame-pointer -g" --extra-cxxflags="-O1 -fno-omit-frame-pointer -g" --extra-ldflags="-L$HOME/ffmpeg_build/include -lubsan" --enable-debug --cc=$CC --cxx=$CXX

else
	./configure --prefix="$HOME/ffmpeg_build" --pkg-config="pkg-config --static" --extra-cflags="-I$HOME/ffmpeg_build/include" --extra-ldflags="-L$HOME/ffmpeg_build/lib" --extra-libs="-lpthread -lm" --bindir="$HOME/bin" --enable-gpl --enable-libass --enable-libfreetype --enable-libmp3lame --enable-libopus --enable-libvorbis --enable-libx264 --enable-libx265 --enable-nonfree  --extra-cflags="-I$HOME/ffmpeg_build/include -O1 -fno-omit-frame-pointer -g" --extra-cxxflags="-O1 -fno-omit-frame-pointer -g" --extra-ldflags="-L$HOME/ffmpeg_build/include -fsanitize=address -fsanitize=undefined -lubsan" --enable-debug --cc=afl-gcc --cxx=afl-g++

fi

make -j
