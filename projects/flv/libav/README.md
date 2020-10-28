## Reference Site
https://github.com/libav/libav

https://stackoverflow.com/questions/62363543/how-to-compile-libav-for-aflgo

## Installation for fuzzing
**download source code**
```
$ git clone https://github.com/libav/libav.git
$ cd libav
```
**setting configure option and make**
```
$ AFL_HARDEN=1 ./configure --cc=afl-gcc
$ AFL_HARDEN=1 make -j4
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_flv/ -o output ./avconv @@
```
