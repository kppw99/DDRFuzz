#!/bin/bash

home=/targets/ffmpeg
seed_home=$home/ddrfuzz_seed
bin=$home/bin/ffmpeg
opt='-m 1000 -t 10000+'

s2s_dir=$seed_home/seq2seq
att_dir=$seed_home/attention
tra_dir=$seed_home/transformer

if [ -z $1 ]; then
	input=$s2s_dir
elif [ $1 == 'attention' ]; then
	input=$att_dir
elif [ $1 == 'transformer' ]; then
	input=$tra_dir
else
	echo ''
	echo './afl-fuzz.sh {input_seed}'
	echo ''
	echo 'Required input seed:'
	echo '  - attention		: attention directory'
	echo '  - transformer	: transformer directory'
	echo '  - default(null)	: origin directory'
	echo ''
	exit 100
fi

output=$home/output
rm -rf $output
echo core >/proc/sys/kernel/core_pattern
timeout -s INT 6h afl-fuzz $opt -i $input -o $output -- $bin -i @@ -f null /dev/null
