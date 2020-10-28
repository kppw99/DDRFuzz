## Reference site
https://hub.docker.com/r/zjuchenyuan/afl

## Installation for fuzzing
**setting CC and CXX path**
```
$ export CC=afl-gcc
$ export CXX=afl-g++
```
**download source code**
```
wget https://sourceforge.net/projects/mp3gain/files/mp3gain/1.6.2/mp3gain-1_6_2-src.zip/download -O mp3gain-1_6_2-src.zipwget https://sourceforge.net/projects/mp3gain/files/mp3gain/1.6.2/mp3gain-1_6_2-src.zip/download -O mp3gain-1_6_2-src.zip
```
**create directory and unzip**
```
$ mkdir -p mp3gain1.6.2 && cd mp3gain1.6.2
$ unzip ../mp3gain-1_6_2-src.zip
```
**setting configure option and make**
```
$ echo "" | sudo tee /proc/sys/kernel/core_pattern # disable generating of core dump file
$ echo 0 | sudo tee /proc/sys/kernel/core_uses_pid
$ echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
$ echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
$ echo 1 | sudo tee /proc/sys/kernel/sched_child_runs_first # tfuzz require this
$ echo 0 | sudo tee /proc/sys/kernel/randomize_va_space # vuzzer require this
$ make
```
**fuzzing with AFL**
```
$ afl-fuzz -i [seed] -o [output] -- ./mp3gain @@
```
