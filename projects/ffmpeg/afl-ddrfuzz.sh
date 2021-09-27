#!/bin/bash

home=/targets/ffmpeg
seed_home=$home/valuable_seed
bin=$home/bin/ffmpeg
opt='-m 1000 -t 10000+'

s2s_dir=$seed_home/seq2seq
att_dir=$seed_home/attention
tra_dir=$seed_home/transformer

if [ $1 == 's2s' ]; then
	input=$s2s_dir
elif [ $1 == 'att' ]; then
	input=$att_dir
elif [ $1 == 'tra' ]; then
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
echo core >/proc/sys/kernel/core_pattern

input1=$input/75
input2=$input/80
input3=$input/85
input4=$input/90

output1=$output+$1+75
output2=$output+$1+80
output3=$output+$1+85
output4=$output+$1+90

rm -rf $output1
timeout -s INT 6h afl-fuzz $opt -i $input1 -o $output1 -- $bin -i @@ -f null /dev/null
rm -rf $output2
timeout -s INT 6h afl-fuzz $opt -i $input2 -o $output2 -- $bin -i @@ -f null /dev/null
rm -rf $output3
timeout -s INT 6h afl-fuzz $opt -i $input3 -o $output3 -- $bin -i @@ -f null /dev/null
rm -rf $output4
timeout -s INT 6h afl-fuzz $opt -i $input4 -o $output4 -- $bin -i @@ -f null /dev/null
