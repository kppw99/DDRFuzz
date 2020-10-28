## Reference Site
https://github.com/pts/sam2p

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
$ git clone https://github.com/pts/sam2p.git
$ cd sam2p
```
**setting configure option and make**
```
$ AFL_HARDEN=1 ./configure
$ AFL_HARDEN=1 make -j4
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_bmp/ -o output ./sam2p @@
```
