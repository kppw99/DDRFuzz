#!/bin/bash

NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
#target_gcno="/targets/libpng/source/pngtest.gcno"
cov_path="/targets/libpng/cov"

llvm-cov-6.0 gcov -b $(find /targets/libpng/source -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt

#for target_gcno in $(find /targets/libpng/source -name "*.gcno")
#do
#	llvm-cov-6.0 gcov -b $target_gcno 2> /dev/null >> $cov_path/$NowDate.txt
#done




