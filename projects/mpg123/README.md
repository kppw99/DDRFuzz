## How to build mpg123
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t mpg123/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --privileged mpg123/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
