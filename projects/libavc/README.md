## How to build libavc
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t libavc/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --privileged libavc/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
