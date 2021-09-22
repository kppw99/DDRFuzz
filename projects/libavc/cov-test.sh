
NowDate=$(date +%m)-$(date +%d)_$(date +%H)_$(date +%M)
cov_path="/targets/libavc/cov"

gcov -b $(find /tool/libavc -name "*.gcno") 2> /dev/null >> $cov_path/$NowDate.txt

