# Functions to manage conversion from m4a to wav.
# Kyle Roth             2018-11-21

from glob import iglob
import os


def remove_m4a(root, verbose=False):
    """Removes all .m4a files as long as a corresponding .wav file is present."""
    for filename in iglob(os.path.join(root, '**/*.m4a'), recursive=True):
        if os.path.isfile(filename[:-3] + 'wav'):
            if verbose:
                print('removing {}'.format(filename))
            os.remove(filename)


def rename(root, verbose=False):
    """Renames .m4a.wav files to .wav."""
    for filename in iglob(os.path.join(root, '**/*.m4a.wav'), recursive=True):
        if verbose:
            print('renaming {}'.format(filename))
        os.rename(filename, filename[:-7] + 'wav')


def cleanup(root, verbose=False):
    """Renames files converted to .wav, and removes original .m4a files. Should be called after running ffmpeg as in
    `voxceleb.sh`."""
    rename(root, verbose)
    remove_m4a(root, verbose)
