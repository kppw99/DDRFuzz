## Reference Site
https://github.com/noirotm/flvmeta/blob/master/INSTALL.md

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
$ git clone https://github.com/noirotm/flvmeta.git
$ cd flvmeta
```
**setting configure option and make**
```
$ cmake .
$ make
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_flv/ -o output .src/flvmeta @@
```
