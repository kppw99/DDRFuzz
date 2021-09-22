#!/bin/bash

NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
cov_path="/targets/libavif/cov"

gcov -b $(find /tool/libavif -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt
