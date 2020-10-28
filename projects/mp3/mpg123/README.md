## Reference Site
https://github.com/georgi/mpg123/blob/master/INSTALL

https://sourceforge.net/p/mpg123/bugs/255/

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
https://www.mpg123.de/snapshot
```
**create directory and unzip**
```
$ tar -xvf sanpshot
$ mv mpg123-20201027022201/ mpg123
$ cd mpg123/build
```
**setting configure option and make**
```
$ CFLAGS="-Os -s" ../configure --with-cpu=generic  --disable-id3v2 --disable-lfs-alias --disable-feature-report --with-seektable=0 --disable-16bit --disable-32bit --disable-8bit --disable-messages --disable-feeder --disable-ntom --disable-downsample --disable-icy --disable-shared
$ make
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_mp3/ -o output -- ./mpg123 -w /dev/null @@
```
