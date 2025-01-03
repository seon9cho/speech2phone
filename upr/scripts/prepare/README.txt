##############################
######## Data Project ########
##############################

Seong-Eun Cho
11/14/2018

Final data project for ACME. Objective is to make an IPA transcriber using audio data.
Data is gathered and cleaned from the General Conference audios in LDS.org

AUDIO_DIR=/Volumes/Seong-Eun_HardDrive/ACME_final/gen_conf_audio

********************* Scrape *********************

# Scrape General Conference data
python script/prepare/gen_conf_scraper.py \
     -o $AUDIO_DIR/raw \
     -y 2016 2018 \
     -s

********************* Clean *********************

# Convert mp3 to wav and resample to 16khz
script/prepare/convert_to_wav.sh $AUDIO_DIR/raw $AUDIO_DIR/16k-wav

# Split long audio data into smaller ~5 second chunks
python script/prepare/split_audio.py \
    -i $AUDIO_DIR/16k-wav \
    -o $AUDIO_DIR/16k-split \
    -n 50 \
    --time-frame 0.2 \
    --min-len 2 \
    --stride 10