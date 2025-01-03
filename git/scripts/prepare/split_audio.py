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
import pywt


def get_properties(filename):
    """Gets the properties of the file at `filename` listed by running `sox --info [filename]."""
    print(os.popen('sox --info {}'.format(filename)).read())


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
        """
        self.rate = rate
        self.samples = samples
        self.length = len(samples)/self.rate
        self.bound = np.abs(samples).mean() * 1.5

    def plot(self, t1=0, t2="end", mode="raw", wavelet_idx=None, ax=None):
        """Plot and display the graph of the sound wave.
        
        `mode` specifies the type of operation to perform before plotting. Possible modes include:
            - 'raw': no operation
            - 'dft': discrete Fourier transform
            - 'spectrum': log of the DFT
            - 'spectrogram': log of the absolute value of the windowed DFT
            - 'cepstrum': IDFT of the spectrogram
            - '[wavelet]': the wavelet reconstruction of the audio by any wavelet name allowed by the pywt package. If a
                           wavelet is specified, `wavelet_idx` must be an integer representing the level of detail to
                           drop.
        """
        plt.figure()
        if ax is None:
            ax = plt.gca()

        if t2 == "end":
            t2 = self.samples.size/self.rate

        if mode == "raw":
            sample_to_plot = self.time_interval(t1, t2)
            t = np.linspace(t1, t2, sample_to_plot.size)
            ax.plot(t, sample_to_plot)
            ax.set_xlabel("Time (s)")
        elif mode == 'dft':
            dft = np.abs(sp.fft(self.time_interval(t1, t2)))
            N = dft.shape[0]
            x_vals = np.arange(1, N // 2 + 1) * self.rate / N
            ax.plot(x_vals, dft[:N//2])
            ax.set_xlabel("Frequency (Hz)")
        elif mode == 'spectrum':
            dft = np.abs(sp.fft(self.time_interval(t1, t2)))
            N = dft.shape[0]
            x_vals = np.arange(1, N // 2 + 1) * self.rate / N
            ax.plot(x_vals, np.log(dft[:N//2]))
            ax.set_xlabel("Frequency (Hz)")
        elif mode == 'spectrogram':
            # TODO: figure out how to plot with xticks set to 1s / 0.01.
            data = self.spectrogram(t1=t1, t2=t2).T
            N = self.rate * 0.03
            y = np.linspace(1, N // 2, N // 2) * self.rate / N
            x = np.linspace(0, len(self.time_interval(t1, t2)) / self.rate, data.shape[1])
            X, Y = np.meshgrid(x, y)
            plt.pcolormesh(X, Y, data, cmap='magma')
            plt.colorbar()
            ax.set_xlabel("Time (s)")
        elif mode == "cepstrum":
            # TODO: figure out why it's plotting black
            dft_sqr = sp.fft(self.time_interval(t1, t2))**2
            spectrum_sqr = np.log(dft_sqr)
            power_ceps = sp.ifft(spectrum_sqr)**2
            print(power_ceps)
            N = dft_sqr.shape[0]
            x_vals = np.arange(1, N // 2 + 1) * self.rate / N
            ax.plot(x_vals, power_ceps[:N//2])
            ax.set_xlabel("Frequency (Hz)")
        else:
            # try doing wavelets, e.g. 'db4'
            try:
                coeffs = self.dwt(t1, t2, w=mode)
                if wavelet_idx is None:
                    raise ValueError('wavelet_idx must be specified for wavelet plot')
                ax.plot(pywt.waverec(coeffs[:-wavelet_idx] + [None] * wavelet_idx, pywt.Wavelet(mode)))
                ax.set_xlabel('Time')
                ax.set_ylabel('Coefficients')
                plt.title('Reconstruction without last {} detail levels'.format(wavelet_idx))
            except ValueError as e:
                # If the error was that the wavelet name was unrecognized, say the mode was set wrong.
                if str(e).startswith('Unknown wavelet name'):
                    raise ValueError("mode must be raw, dft, spectrum, cepstrum, or a wavelet name (e.g. 'db4')")
                else:  # There was a different error, so raise it.
                    raise

    def export(self, filename):
        """Generate a wav file called filename from the sample rate and samples. If the array of samples is not of type
        int16, scale it so that it is."""
        samples = np.int16(self.samples)
        directory = '/'.join(filename.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        wavfile.write(filename, self.rate, samples)

    def __add__(self, other):
        """Add two sound waves together into one wave."""
        if self.samples.size != other.samples.size:
            raise ValueError("sample size of the two objects are not equal")
        return SoundWave(self.rate, self.samples + other.samples)

    def append(self, other):
        """Append additional samples to the end of the current samples."""
        if self.rate != other.rate:
            raise ValueError("sample rates of the two objects are not equal")
        self.samples = np.hstack([self.samples, other.samples])

    def time_interval(self, t1, t2):
        if t2 == 'end':
            t2 = self.samples.size / self.rate
        return self.samples[int(t1 * self.rate):int(t2 * self.rate)]

    def split(self, time_frame=0.2, min_len=5, stride=10):
        """Split the audio into small samples. Returns a list of SoundWave objects."""
        min_len *= self.rate
        window = int(self.rate * time_frame)
        pause = False
        start = []
        stop = []
        # Move across the audio, skipping every `stride` entries.
        loop = tqdm(total=len(self.samples) // stride -
                    window, position=0, leave=False)
        for i in range(len(self.samples) // stride - window):
            # i is the number of the stride, and j is the actual index in the audio.
            j = i * stride
            if not pause:
                if np.abs(self.samples[j:j + window]).max() <= self.bound:
                    start.append(j)
                    # We've encountered a window where the max is less than the bound. Keep looking until we find a
                    # window where that's no longer true.
                    pause = True
            else:
                if np.abs(self.samples[j:j + window]).max() > self.bound:
                    stop.append(j+window)
                    # We've found the end of the pause.
                    pause = False
            loop.update(1)
        loop.close()

        if len(start) > len(stop):
            # If the pause began but the end of the file was encountered.
            stop.append(len(self.samples) - 1)

        # Find the mean of each pair, which will be the cut position.
        ti = np.vstack([start, stop]).mean(axis=0).astype(int)
        sample_splits = []
        left = ti[0]
        for i in range(len(ti) - 1):
            current_sample = SoundWave(self.rate, self.samples[left:ti[i+1]])
            sample_splits.append(current_sample)
            left = ti[i+1]
        return sample_splits

    def dft(self, time_frame=0.03, overlap_frame=0.01, t1=0, t2='end'):
        """Calculates the DFT of the audio."""
        if t2 == "end":
            t2 = self.samples.size/self.rate
        
        window = int(self.rate * time_frame)
        stride = int(self.rate * overlap_frame)

        # pull only the samples in the specified interval
        samples = self.time_interval(t1, t2)

        i = 0
        s = []
        while True:
            j = i + window
            if j >= len(samples):
                break
            dft = sp.fft(samples[i:j])
            s.append(dft[:window//2])
            i += stride

        return np.real(np.array(s))
    
    def spectrogram(self, time_frame=0.03, overlap_frame=0.01, t1=0, t2='end'):
        """Calculates the spectrogram of the audio."""
        return np.log(np.abs(self.dft(time_frame, overlap_frame, t1, t2)))
    
    def cepstrum(self, time_frame=0.03, overlap_frame=0.01, t1=0, t2='end'):
        """Calculates the cepstrum (IFFT of the spectrogram) of the audio."""
        return np.real(sp.ifft(self.spectrogram(time_frame, overlap_frame, t1, t2)))

    def dwt(self, start=0, end='end', w='db4'):
        """Computes the discrete wavelet transform of the audio."""
        w = pywt.Wavelet(w)
        return pywt.wavedec(self.time_interval(start, end), w)


def split_audio(audio, output_dir, audio_num, lang, tf, ml, stride):
    """Split and export the audio to `output_dir`. Calls the SoundWave's split function. Works using a specific
    language. Needs to be updated to generalize to all audio."""
    rate, wave = wavfile.read(audio)
    S = SoundWave(rate, wave)
    splits = S.split(time_frame=tf, min_len=ml, stride=stride)
    for i, s in enumerate(splits):
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
        if len(audio_list) <= n:
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
    parser.add_argument(
        '-o', '--output', help="Output directory", required=True)
    parser.add_argument('-n', '--number', type=int,
                        help="Number of audio samples to process per language."
                             "If not defined, process all files.")
    parser.add_argument('--time-frame', type=float,
                        help="Window of pause to detect as pause (seconds)")
    parser.add_argument('--min-len', type=float,
                        help="Minimum length a splitted audio can take (seconds)")
    parser.add_argument('--stride', type=int,
                        help="Number of samples to skip for the sliding window"
                             "(Useful for speeding up the process)")
    args = parser.parse_args()

    main(args.input, args.output, args.number,
         args.time_frame, args.min_len, args.stride)
