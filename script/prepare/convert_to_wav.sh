#!/bin/bash

if [ -z "$1" ]; then
    echo "requires input directory as the first argument"
    echo "./convert_to_wav.sh [INPUT_DIR] [OUTPUT_DIR]"
    exit 1
fi

if [ -z "$2" ]; then
	echo "requires output directory as the second argument"
	echo "./convert_to_wav.sh [INPUT_DIR] [OUTPUT_DIR]"
	exit 1
fi

AUDIO_DIR=$1
OUT_DIR=$2
FOLDERS="$(ls $AUDIO_DIR)"

for folder in $FOLDERS; do
    if [ -d "$OUT_DIR/$folder" ]; then
        continue
    fi
    mkdir $OUT_DIR/$folder
    FILES="$(ls $AUDIO_DIR/$folder)"
    for file in $FILES; do
        audio_name=`echo $file | cut -d '.' -f 1`
        extension=`echo $file | rev | cut -d '.' -f 1 | rev`
        sox $AUDIO_DIR/$folder/$file -r 16000 $OUT_DIR/$folder/$audio_name.wav
    done
done