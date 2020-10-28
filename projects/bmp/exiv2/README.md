## Reference Site
https://gist.github.com/yevgenypats/c939b165321260f1ef05774be2b6a017

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
$ git clone https://github.com/Exiv2/exiv2.git
$ cd exiv2
```
**setting configure option and make**
```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ cmake --build .
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_bmp/ -o output ./exiv2 @@
```
