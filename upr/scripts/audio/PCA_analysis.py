import audio
from scipy.io import wavfile
import os
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PCA:
    def __init__(self, X, y=None, s=None):
        self.X = X - X.mean(axis=0)
        self.y = y
        if s == None:
            n, d = X.shape
            s = d
        U, sig, Vh = np.linalg.svd(self.X)
        self.sig = sig[:s]**2
        self.Vh = Vh[:s]
        self.a = self.transform(X)
        self.proj_X = self.project(X)
    
    def transform(self, x):
        return self.Vh@x.T
    
    def project(self, x):
        return self.Vh.T@self.a

def get_phoneme_data(phonemes=['f', 's', 'g', 't', 'p', 'schwa', 'ae', 'i', 'a']):
    phone_data = []
    phoneme_dict = {}
    for i, p in enumerate(phonemes):
        path = "../../ipa/"
        for f in os.listdir(path + p):
            fname = path + p + '/' + f
            rate, sample = wavfile.read(fname)
            n = len(sample)
            wave_spec = np.log(np.abs(sp.fft(sample))[:n//2])
            wave_spec = signal.resample(wave_spec, 720)
            phoneme_dict[i] = p
            phone_data.append(np.hstack([i, wave_spec]))

    phone_data = np.array(phone_data)
    X = phone_data[:, 1:]
    y = phone_data[:, 0]
    return X, y, phoneme_dict


X, y, phoneme_dict = get_phoneme_data()
print(X.shape)
pca = PCA(X)
n, d = X.shape
print(pca.sig)

fig = plt.figure()
plt.plot(np.arange(n), pca.sig)
plt.xlabel("number of components")
plt.ylabel("explained variance")
plt.title("Phoneme data scree plot")
plt.show()

pca2 = PCA(X, y=y, s=2)
A = np.vstack([pca2.a, pca2.y])
for i in np.unique(y):
    curr_a = A.T[A[2] == i][:, :2]
    plt.scatter(curr_a[:, 0], curr_a[:, 1], label=phoneme_dict[i], alpha=0.7)
plt.legend()
plt.show()

i = 1
p = 0
while p < .9:
    p = np.sum(pca.sig[:i]) / np.sum(pca.sig)
    print(i, p)
    i+=1