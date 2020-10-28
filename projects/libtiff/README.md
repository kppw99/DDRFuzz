## Reference Site
https://bugs.launchpad.net/ubuntu/+source/tiff/+bug/1685451

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
git clone https://github.com/vadz/libtiff.git
cd libtiff
```
**setting configure option and make**
```
$ AFL_HARDEN=1 ./configure --disable-shared
$ AFL_HARDEN=1 make
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_bmp/ -o output -- ./bmp2tiff -c jpeg:r:50 @@
$ afl-fuzz -i seed_bmp/ -o output -- ./bmp2tiff -c none @@
$ afl-fuzz -i seed_bmp/ -o output -- ./bmp2tiff -c /dev/null @@
```
