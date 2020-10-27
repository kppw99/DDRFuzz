## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
wget https://sourceforge.net/projects/mpg321/files/mpg321/0.3.2/mpg321_0.3.2.orig.tar.gz/download
```
**create directory and unzip**
```
$ tar -zxvf download
$ mv mpg321-0.3.2-orig/ mpg321
$ cd mpg321
```
**setting configure option and make**
```
$ sh ../configure
$ make
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_mp3/ -o output -- ./mpg321 --stdout @@
```
