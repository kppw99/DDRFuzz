## How to build ffmpeg
- *Prerequisite: preparation for the base/ddrfuzz:latest container*
```
# Build Dockerfile
$ docker build -t ffmpeg/ddrfuzz:latest . -f Dockerfile

# Execute Container
$ docker run -it --privileged ffmpeg/ddrfuzz:latest /bin/bash

# Optimize Seed (Optional)
$ ./opt-seed.sh

# Fuzzing
$ ./afl-fuzz.sh
```
