#!/bin/bash


NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
cov_path="/targets/libtiff/cov"

gcov -b $(find /targets/libtiff/source -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt

