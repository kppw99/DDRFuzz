#!/bin/bash

home=/targets/ffmpeg
seed_home=$home/seed
bin=$home/bin/ffmpeg
opt='-m 1000 -t 10000+'

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

output=$home/output
echo core >/proc/sys/kernel/core_pattern

input1=$input
input2=$home/valuable_seed/wgan

output1=$home/output+origin
output2=$home/output+wgan

rm -rf $output1
timeout -s INT 6h afl-fuzz $opt -i $input1 -o $output1 -- $bin -i @@ -f null /dev/null
rm -rf $output2
timeout -s INT 6h afl-fuzz $opt -i $input2 -o $output2 -- $bin -i @@ -f null /dev/null
