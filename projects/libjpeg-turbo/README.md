## How to build libjpeg-turbo
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t libjpeg-turbo/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --privileged libjpeg-turbo/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
