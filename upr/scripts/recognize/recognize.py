# Module for recognizing phonemes in audio data.
# Built by Kyle Roth. 2018-12-10.

from scripts.prepare.split_audio import SoundWave
from scripts.utils import func_compile

import numpy as np
from scipy import linalg as la
from warnings import warn
from os import listdir, path

from scipy import signal
from scipy.io import wavfile
from scipy import fft


def get_phoneme_dict(root_dir):
    """Get phonemes from the specified directory, returning a dictionary mapping phoneme characters to lists of
    SoundWave files.
    
    In the specified directory, each phoneme should be specified by a subdirectory labeled with the phoneme character.
    The subdirectory should contain .wav files with instances of the phoneme.
    """
    # get immediate subdirectories
    phoneme_chars = [name for name in listdir(root_dir)
                        if path.isdir(path.join(root_dir, name)) and name != '16k-wav']
    
    phoneme_dict = {}

    for char in phoneme_chars:
        phoneme_dict[char] = []
        for name in listdir(path.join(root_dir, char)):
            rate, data = wavfile.read(path.join(root_dir, char, name))
            phoneme_dict[char].append(SoundWave(rate, data))
    
    return phoneme_dict


def l2_metric(a, b):
    """Defines the standard L2 distance metric between a and b: ||a - b||_2."""
    return np.linalg.norm(a - b)


class PhonemeEmbedding(object):
    """Attempts phoneme recognition using an arbitrary metric, comparing given audio data with known samples of phonemes
    to find the closest match.
    """

    def __init__(self, phoneme_dict, embedding=SoundWave.cepstrum):
        """Initializes the embedding space with the given phonemes."""
        # this will be a dictionary just like `phoneme_dict`, but mapping character strings to the features created by
        # the embedding function
        self.space = {}
        self.embedding = embedding
        self.add_embeddings(phoneme_dict)
    
    def add_embeddings(self, phoneme_dict):
        """Places the audio in `phoneme_dict` into the chosen embedding space.
        
        `phoneme_dict` should be a dictionary mapping from phoneme character strings to lists of SoundWave
        representations.
        """
        # apply the embedding function to each SoundWave and store it in the embedding space
        for char, phones in phoneme_dict:
            for phone in phones:
                if char not in self.space.keys():
                    self.space[char] = []
                self.space[char].append(self.embedding(phone))
    
    def nearest(self, sound, metric=l2_metric):
        """Return the nearest neighbor to the given sound in the embedding space, under the given metric.
        
        The default metric is the standard L2 metric.
        """
        # extract the features of the audio for comparison
        sound_features = self.embedding(sound)

        best, best_dist = np.inf, None

        # find the nearest neighbor
        for char in self.space.keys():
            for embedded_features in self.space[char]:
                current_dist = metric(embedded_features, sound_features)
                if current_dist < best_dist:
                    best = char
                    best_dist = current_dist
        
        return best


class MatrixOperator(object):
    """Create callable object equivalent to multiplying by a matrix."""

    def __init__(self, A):
        """Store the matrix to multiply."""
        self.A = A
        self.Ainv = None
    
    def __call__(self, other):
        """Apply the matrix operator to `other`."""
        return self.A.dot(other)
    
    def inv(self, other):
        """Apply the inverse of the matrix to `other`."""
        # compute A's inverse if not yet calculated
        if self.Ainv is None:
            self.Ainv = np.linalg.inv(self.A)
        
        return self.Ainv.dot(other)


def find_nearest_semidefinite(A):
    """Find the nearest matrix to A under the Frobenius norm
    
    Diagonalize the matrix, and set all negative eigenvalues to zero.
    """
    eigvals, eigvecs = la.eig(A)
    eigvals[eigvals < 0] = 0
    return eigvecs.T.dot(np.diag(eigvals)).dot(eigvecs)


class PhonemeSimilarity(PhonemeEmbedding):
    """Attempts phoneme recognition using the learned embedding space described in the paper cited below.
    
    Xing, et al. "Distance metric learning, with application to clustering with side-information", UC Berkeley, 2003.
    """

    def __init__(self, phoneme_dict, preprocessor=None, alpha=0.1, maxiters=30, tol=1e-4):
        """Learn the embedding operator A that minimizes the distance between similar speech sounds, while ensuring the
        distance between distinct speech sounds is greater than 1.

        The preprocessor is applied to each sound before embedding. The default preprocessor takes the spectrum of audio
        and resamples to 240.
        """
        if preprocessor is None:
            preprocessor = func_compile(fft, np.abs, np.log, (signal.resample, 240))

        # create a list of the preprocessed sounds as well as S and D, from `phoneme_dict`
        vals = []
        S = []
        D = []
        for char in phoneme_dict.keys():
            # preprocessing
            phones = [preprocessor(phone.samples) for phone in phoneme_dict[char]]

            # pair each phone in current phones with a previously seen phone, and add it to D
            for phone in phones:
                for other in vals:
                    D.append((phone, other))

            # add phone pairs to S
            while phones:
                phone = phones[0]
                vals.append(phone)
                del phones[0]

                for other in phones:
                    S.append((phone, other))

        # learn A
        A = self.learn_A(vals, S, D, alpha, maxiters, tol)

        # embed the phonemes with A^{1/2} as the embedding function (see Section 2 of the paper)
        sqrt_A = MatrixOperator(la.sqrtm(A))
        super(PhonemeSimilarity, self).__init__(phoneme_dict, embedding=func_compile(preprocessor, sqrt_A))
    
    def matrix_metric_sq(self, x, y, A):
        """Return ||x - y||_A^2."""
        diff = x - y
        return diff.dot(A).dot(diff)
    
    def matrix_metric(self, x, y, A):
        """Return ||x - y||_A."""
        return np.sqrt(self.matrix_metric_sq(x, y, A))
    
    def learn_A(self, vals, S, D, alpha, maxiters, tol):
        """Use the algorithm in Figure 1 of the paper to find a matrix solution to the following:
        
            Maximize the sum of squared distances between dissimilar objects,
                requiring that the sum of squared distances between similar objects be less than or equal to 1, and
                requiring that A be positive definite.
        
        This is (3-5) from the paper.
        """

        # define g according to the paper (Section 2.2)
        def g(A, D):
            return np.sum([self.matrix_metric(x, y, A) for x, y in D])
        
        # define g's gradient
        def grad_g(A, D):
            return np.sum([np.outer(x - y, x - y) / 2 / np.sqrt((x - y).dot(A).dot(x - y)) for x, y in D])
        
        # define f's gradient
        def grad_f(A, S):
            return np.sum([np.outer(x - y, x - y) / 2 / np.sqrt((x - y).dot(A).dot(x - y)) for x, y in S])

        # start with identity matrix of appropriate size for one (and hopefully all) of the objects
        A = np.eye(len(vals[0]))

        for _ in range(maxiters):
            prev_A = A  # for testing convergence

            # iterative projections to ensure properties (4) and (5)
            for i in range(maxiters):
                prev_proj_A = A  # for testing convergence
                # TODO: closest A where ||x_i - x_j||_A^2 <= 1 for similar x_i,x_j. For now, we include this in the
                # gradient ascent step by maximizing g - f.
                A = find_nearest_semidefinite(A)
                if np.linalg.norm(A - prev_proj_A) < tol:
                    # convergence requirement met
                    break
                elif i + 1 == maxiters:
                    # if this was the last iteration, A did not converge
                    warn('iterative projections did not converge')
            # gradient ascent
            A = A + alpha * (grad_g(A, D) - grad_f(A, S))
            if np.linalg.norm(A - prev_A) < tol:
                # convergence requirement met
                return A
        
        # if A did not converge, return but print a warning
        warn('gradient ascent did not converge')
        return A
