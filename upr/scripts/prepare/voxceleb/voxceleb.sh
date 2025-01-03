#!/bin/bash

# These steps download and unpack the VoxCeleb datasets. The variable $target_dir
# Is the directory where the data will be placed.
# Kyle Roth                   2018-11-17

target_dir=/media/kylrth/KYLEBAK/data_project/data

# Download VoxCeleb1
wget --user voxceleb000 --password vgg \
    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa
wget --user voxceleb000 --password vgg \
    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab
wget --user voxceleb000 --password vgg \
    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac
wget --user voxceleb000 --password vgg \
    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad
wget --user voxceleb000 --password vgg \
    http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip

# Download VoxCeleb2
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacaa
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacab
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacac
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacad
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacae
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacaf
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacag
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_dev_aacah
wget --user voxceleb000 --password vgg \
    http://zeus.robots.ox.ac.uk/voxceleb2/aac/vox2_test_aac.zip

# concatenate the separate dev set files into one zip file for each dev set
cat vox1_dev* > vox1_dev_wav.zip
cat vox2_dev* > vox2_dev_mp4.zip

# move the files into the target directory
mv vox1_dev_wav.zip $target_dir
mv vox1_test_wav.zip $target_dir
mv vox2_dev_mp4.zip $target_dir
mv vox2_test_aac.zip $target_dir

pushd $target_dir

# unzip and give all directories similar descriptive names
unzip vox1_dev_wav.zip
mv vox1_dev_wav vox1_dev
rm vox1_dev_wav.zip

unzip vox1_test_wav.zip
mv vox1_test_wav vox1_test
rm vox1_test_wav.zip

unzip vox2_dev_mp4.zip
mv dev vox2_dev
rm vox2_dev_mp4.zip

unzip vox2_test_aac.zip
mv vox2_test_aac vox2_test
rm vox2_test_aac.zip

mv -v vox2_dev/aac/* vox2_dev/
rm -r vox2_dev/aac/
mv -v vox2_test/aac/* vox2_test/
rm -r vox2_test/aac/

# change all the vox2 files to 16kHz .wav
find vox2_dev/ -name "*.m4a" -exec ffmpeg -loglevel panic -i {} {}.wav \;
# add -print0 after the -name parameter if you want to see progress
find vox2_dev/ -name "*.m4a.wav" -exec rename 's/\.m4a\.wav$/\.wav/' {}

popd
