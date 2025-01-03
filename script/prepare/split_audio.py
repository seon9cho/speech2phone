import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import IPython
import scipy as sp
import time
from scipy import signal
from scipy.misc import imread
import pylab
from tqdm import tqdm
import argparse
import glob
import os

class SoundWave(object):
    """A SoundWave class for working with digital audio signals.
    Attributes:
        rate (int): The sample rate of the sound.
        samples ((n,) ndarray): NumPy array of samples.
    """
    def __init__(self, rate, samples):
        """Set the SoundWave class attributes.
        Parameters:
            rate (int): The sample rate of the sound.
            samples ((n,) ndarray): NumPy array of samples.
        Returns:
            A SoundWave object.
        """
        self.rate = rate
        self.samples = samples
        self.length = len(samples)/self.rate
        self.bound = np.abs(samples).mean() * 2
        
    def plot(self, t1=0, t2="end"):
        """Plot and display the graph of the sound wave."""
        if t2 == "end":
            t2 = self.samples.size/self.rate
        sample_to_plot = self.samples[int(t1*self.rate):int(t2*self.rate)]
        t = np.linspace(t1, t2, sample_to_plot.size)
        plt.plot(t, sample_to_plot)
        plt.xlabel("Time (s)")
        plt.show()
        
    def export(self, filename):
        """Generate a wav file called filename from the sample rate and samples. 
        If the array of samples is not of type int16, scale it so that it is."""
        samples = np.int16(self.samples)
        wavfile.write(filename, self.rate, samples)
        
    def __add__(self, other):
        """Add two sound waves together into one wave."""
        if self.samples.size != other.samples.size:
            raise ValueError("Sample size of the two objects are not equal.")
        return SoundWave(self.rate, self.samples + other.samples)
        
    def append(self, other):
        """Append additional samples to the end of the current samples."""
        if self.rate != other.rate:
            raise ValueError("Sample rates of the two objects are not equal.")
        self.samples = np.hstack([self.samples, other.samples])

    def time_interval(self, t1, t2):
        out_sample = self.samples[int(t1*self.rate):int(t2*self.rate)]
        return out_sample
    
    def split(self, time_frame=0.2, min_len=5, stride=10):
        """Split the aduio into small samples
        Returns list of SoundWave object"""
        min_len*=self.rate
        window = int(self.rate * time_frame)
        pause = False
        start = []
        stop = []
        loop = tqdm(total=len(self.samples)//stride - window, position=0, leave=False)
        for i in range(len(self.samples)//stride - window):
            j = i*stride
            if not pause:
                if np.abs(self.samples[j:j+window]).max() <= self.bound:
                    start.append(j)
                    pause = True
            else:
                if np.abs(self.samples[j:j+window]).max() > self.bound:
                    stop.append(j+window)
                    pause = False
            loop.update(1)
        loop.close()
        if len(start) > len(stop):
            stop.append(len(self.samples) - 1)
        
        ti = np.vstack([start, stop]).mean(axis=0).astype(int)
        sample_splits = []
        left = ti[0]
        for i in range(len(ti) - 1):
            sample_len = ti[i + 1] - left
            if sample_len < min_len:
                continue
                print("bad", left, ti[i+1])
            else:
                current_sample = SoundWave(self.rate, self.samples[left:ti[i+1]])
                sample_splits.append(current_sample)
                left = ti[i+1]
        return sample_splits
    
    def plot_dft(self):
        """Take the DFT of the sound wave. Scale the x-axis so the x-values correspond
        to the frequencies present in the plot. Display the left half of the plot."""
        dft = abs(sp.fft(self.samples))
        N = dft.shape[0]
        x_vals = np.linspace(1, N, N)
        x_vals = x_vals * self.rate / N
        plt.plot(x_vals[:N//2], dft[:N//2])
        plt.xlabel("Frequency (Hz)")
        plt.show()
    
    def plot_spectrogram(self, t1=0, t2="end"):
        if t2=="end":
            t2 = self.samples.size/self.rate
        sample_to_plot = self.samples[int(t1*self.rate):int(t2*self.rate)]
        plt.specgram(sample_to_plot,Fs=S.rate)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

def split_audio(audio, output_dir, audio_num, lang, tf, ml, stride):
    rate, wave = wavfile.read(audio)
    S = SoundWave(rate, wave)
    splits = S.split(time_frame=tf, min_len=ml, stride=stride)
    for i,s in enumerate(splits):
        if i < 2:
            continue
        if len(s.samples) / s.rate > 15:
            continue
        file_name = lang + '_' + audio_num + '_' + format(i, '03d') + '.wav'
        s.export(output_dir + '/' + lang + '/' + file_name)

def main(input_dir, output_dir, n, tf, ml, stride):
    if n is None:
        n = np.inf
    if tf is None:
        tf = 0.2
    if ml is None:
        ml = 5
    if stride is None:
        stride = 10

    dir_list = os.listdir(input_dir)
    for i, lang in enumerate(dir_list):
        # Skip if not directory
        if not os.path.isdir(input_dir + '/' + lang):
            continue
        audio_list = glob.glob(input_dir + '/' + lang + "/*")
        # Skip if directory is empty
        if len(audio_list) == 0:
            continue
        # Create output directory
        if not os.path.isdir(output_dir + '/' + lang):
            os.mkdir(output_dir + '/' + lang)
        print("Current language: {}".format(lang))
        print("{}/{}".format(i, len(dir_list)))
        # Condition for whether there are n+ audios in the directory
        if len(audio_list) <= n :
            for i, audio in enumerate(audio_list):
                print("Audio {}/{}".format(i, len(audio_list)))
                audio_num = format(i, '03d')
                split_audio(audio, output_dir, audio_num, lang, tf, ml, stride)
        else:
            rand_index = np.random.randint(0, len(audio_list), n)
            n_audio_list = [audio_list[i] for i in rand_index]
            for i, audio in enumerate(n_audio_list):
                print("Audio {}/{}".format(i, len(n_audio_list)))
                audio_num = format(i, '03d')
                split_audio(audio, output_dir, audio_num, lang, tf, ml, stride)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description="Split long audio into ~5 second samples")
    parser.add_argument('-i', '--input', help="Input directory", required=True)
    parser.add_argument('-o', '--output', help="Output directory", required=True)
    parser.add_argument('-n', '--number', type=int, 
                        help="Number of audio samples to process per language." \
                             "If not defined, process all files.")
    parser.add_argument('--time-frame', type=float,
                        help="Window of pause to detect as pause (seconds)")
    parser.add_argument('--min-len', type=float,
                        help="Minimum length a splitted audio can take (seconds)")
    parser.add_argument('--stride', type=int,
                        help="Number of samples to skip for the sliding window" \
                             "(Useful for speeding up the process)")
    args = parser.parse_args()

    main(args.input, args.output, args.number, 
        args.time_frame, args.min_len, args.stride)


