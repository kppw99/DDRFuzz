## Reference Site
https://github.com/kynesim/tstools

## Installation for fuzzing
**download source code**
```
$ git clone https://github.com/kynesim/tstools.git
$ cd tstools
```
**modify Makefile**
```
# ifdef CROSS_COMPILE
# CC = $(CROSS_COMPILE)gcc
# else
# CC = gcc
# endif
CC = afl-fuzz
```
**setting configure option and make**
```
$ make
```
**fuzzing with AFL**
```
$ afl-fuzz -i ~/seed_flv/ -o output ./bin/ps2ts @@
```
