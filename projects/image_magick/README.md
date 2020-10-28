## Reference Site
https://github.com/moshekaplan/FuzzImageMagick

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
git clone https://github.com/ImageMagick/ImageMagick.git
```
**setting configure option and make**
```
$ AFL_HARDEN=1 ./configure --with-bzlib=no --with-djvu=no --with-dps=no --with-fftw=no --with-fpx=no --with-fontconfig=no --with-freetype=no --with-gvc=no --with-jbig=no --with-jpeg=no --with-lcms=no --with-lqr=no --with-lzma=no --with-openexr=no --with-openjp2=no --with-pango=no --with-png=no --with-tiff=no --with-raqm=no --with-webp=no --with-wmf=no --with-x=no --with-xml=no --with-zlib=no --enable-hdri=no --disable-shared
$ AFL_HARDEN=1 make
```
**fuzzing with AFL**
```
$ afl-fuzz -i seed_bmp/ -o output -- ./magick /dev/null @@
```
