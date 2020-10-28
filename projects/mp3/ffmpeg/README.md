## Reference Site
https://hardik05.wordpress.com/2020/08/22/fuzzing-ffmpeg-with-afl-on-ubuntu/

## Installation for fuzzing
**setting CC and CCX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
$ git clone https://github.com/gypified/libmpg123.git
$ cd libmpg123
```
**setting configure option and make**
```
$ ./configure --prefix="$HOME/ffmpeg_build" --pkg-config="pkg-config --static" --extra-cflags="-I$HOME/ffmpeg_build/include" --extra-ldflags="-L$HOME/ffmpeg_build/lib" --extra-libs="-lpthread -lm" --bindir="$HOME/bin" --enable-gpl --enable-libass --enable-libfreetype --enable-libmp3lame --enable-libopus --enable-libvorbis --enable-libx264 --enable-libx265 --enable-nonfree --cc=afl-clang --cxx=afl-clang++ --extra-cflags="-I$HOME/ffmpeg_build/include -O1 -fno-omit-frame-pointer -g" --extra-cxxflags="-O1 -fno-omit-frame-pointer -g" --extra-ldflags="-L$HOME/ffmpeg_build/include -fsanitize=address -fsanitize=undefined -lubsan" --enable-debug --disable-x86asm
$ AFL_HARDEN=1 make -j8
```
**fuzzing with AFL**
```
$ afl-cmin -i input -o mininput -- ./ffmpeg -i @@ test
$ afl-fuzz -i seed_mp3/ -o fuzz -m none -- ./ffmpeg -i @@ test
```
