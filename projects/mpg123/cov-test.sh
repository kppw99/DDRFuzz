#!/bin/bash

NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
cov_path="/targets/mpg123/cov"

llvm-cov-6.0 gcov -b $(find /targets/mpg123/source/build -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt


