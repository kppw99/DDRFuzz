#!/bin/bash

NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
cov_path="/targets/libspng/cov"

gcov -b $(find /targets/libspng/source -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt


#for target_gcno in $(find /targets/libspng/source -name "*.gcno")
#do
#        gcov -b $target_gcno 2> /dev/null >> $cov_path/$NowDate.txt
#done

