## How to build ImageMagick
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t imagemagick/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --privileged imagemagick/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
