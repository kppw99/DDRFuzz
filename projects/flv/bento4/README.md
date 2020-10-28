## Reference Site
https://github.com/axiomatic-systems/Bento4

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
$ git clone https://github.com/axiomatic-systems/Bento4.git
$ cd Bento4
```
**setting configure option and make**
```
$ mkdir cmakebuild
$ cd cmakebuild
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```
**fuzzing with AFL**
```
$ afl-fuzz -i ~/seed_flv/ -o output ./mp42aac @@
```
