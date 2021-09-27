#!/bin/bash

home=/targets/mpg123
seed_home=$home/seed
bin=$home/bin/mpg123
opt='-m 100 -t 1000'

origin_dir=$seed_home/org
copt_dir=$seed_home/copt
topt_dir=$seed_home/topt

if [ -z $1 ]; then
	input=$origin_dir
elif [ $1 == 'cmin' ]; then
	input=$copt_dir
elif [ $1 == 'tmin' ]; then
	input=$topt_dir
else
	echo ''
	echo './afl-fuzz.sh {input_seed}'
	echo ''
	echo 'Required input seed:'
	echo '  - cmin		: cmin directory'
	echo '  - tmin		: tmin directory'
	echo '  - default(null)	: origin directory'
	echo ''
	exit 100
fi

input1=$input
input2=$home/valuable_seed/wgan

output1=$home/output+origin
output2=$home/output+wgan
echo core >/proc/sys/kernel/core_pattern

rm -rf $output1
timeout -s INT 6h afl-fuzz $opt -i $input1 -o $output1 -- $bin -w /dev/null @@ 

rm -rf $output2
timeout -s INT 6h afl-fuzz $opt -i $input2 -o $output2 -- $bin -w /dev/null @@ 
