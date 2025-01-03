# Functions to clean the rough edges from audio.
# Kyle Roth                     2018-11-16

from scripts.prepare.split_audio import SoundWave
from scipy.io import wavfile
import numpy as np

# To run against all the data in a directory, run something like the following:
#
# paths = glob('data/*_wav/**/*.wav', recursive=True)
# for path in tqdm(paths):
#    new_path = 'clean' + path[4:]
#
#    # cut off edges where a speech segment is less than half a second
#    ess = edges.fix(path, burst_length=500, pause_length=200)
#    ess.export(new_path)
#

def fix(loc, burst_length, pause_length):
    """Reads the audio file found at loc, and cuts off the edges if they contain a segment of speech shorter than
    burst_length. burst_length should be given in milliseconds."""
    rate, wave = wavfile.read(loc)

    # convert to intervals
    burst_interval = int(burst_length * rate / 1000)
    pause_interval = int(pause_length * rate / 1000)

    # set the bound
    bound = np.abs(wave).mean()

    # This will be where the beginning should be cut off. If negative, no cut will occur.
    cut_idx = -1

    # Cut in the middle of the pause, if one is found. Only check 100 frames per pause_length
    for pause_start_idx in range(0, burst_interval, pause_interval // 100 + 1):
        if np.all(np.abs(wave[pause_start_idx:pause_start_idx + pause_interval]) < bound):
            # Every sample in this window is less than the bound, so we found the window we want.
            cut_idx = pause_start_idx + pause_interval // 2
            break
    
    if cut_idx > 0:
        # extra.append(SoundWave(rate, wave[:cut_idx]))
        wave = wave[cut_idx:]
    
    # where the end should be cut off
    cut_idx = 1

    for pause_end_idx in range(0, burst_interval, pause_interval // 100 + 1):
        if pause_end_idx == 0:  # Python does weird things ending a slice at zero.
            if np.all(np.abs(wave[-pause_interval:]) < bound):
                cut_idx = -pause_interval // 2
                break
        else:
            if np.all(np.abs(wave[-pause_end_idx - pause_interval:-pause_end_idx]) < bound):
                cut_idx = -pause_end_idx - pause_interval // 2
                break

    if cut_idx < 0:
        # extra.append(SoundWave(rate, wave[cut_idx:]))
        wave = wave[:cut_idx]
    
    return SoundWave(rate, wave)  #, extra
