## How to build Libmpeg2
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t libpng/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --name libpng --privileged libpng/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
