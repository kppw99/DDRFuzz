## How to build libtiff
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t libtiff/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --privileged libtiff/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
