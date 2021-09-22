#!/bin/bash

NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
cov_path="/targets/ffmpeg/cov"

llvm-cov-6.0 gcov -b $(find /targets/ffmpeg/source -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt


