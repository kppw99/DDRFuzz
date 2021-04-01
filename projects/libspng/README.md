## How to build Libspng
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t libspng/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --name libspng --privileged libspng/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
