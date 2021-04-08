## How to build libavif
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t libavif/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --privileged libavif/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
