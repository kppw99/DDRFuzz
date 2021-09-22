#!/bin/bash

NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
cov_path="/targets/libjpeg-turbo/cov"

gcov -b $(find /tool/libjpeg-turbo -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt




