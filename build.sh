#!/bin/bash

if [ -z $1 ]; then
    echo ''
    echo './build.sh [project name]'
    echo ''
    exit 100
else
	function build {
		docker-compose build ddrfuzz
		docker-compose build $1
	}

	build "$@"
fi

# ./build.sh mpg123

