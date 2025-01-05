# *speech2phone*: Phoneme Segmentation and Classification from Raw Audio

*We extend the speech recognition problem by converting raw audio data into discrete phonemes, rather than directly into language. For classifying phonemes, we combine several classical machine learning techniques such as dimensionality reduction with the expressive power of deep neural nets. We compare recurrent and convolutional neural networks and various embeddings, achieving 74.9\% on the TIMIT dataset. We also attempt phoneme segmentation, getting an average error of 9.398 ms with precision 70.65\% and recall 88.62\% using a reinforcement learning model.*

## Introduction

Speech recognition is a fundamental discipline of computational linguistics. It has traditionally been a very difficult problem to solve even with statistical models but has recently seen marked improvement through the use of neural networks.

Similar to speech recognition, phoneme recognition is the challenge of transcribing audio input into a phonetic alphabet, e.g. the International Phonetic Alphabet (IPA). While a language's regular alphabet and spelling are decided by history and don't necessarily match pronunciation, the goal of a phonetic alphabet is to represent exactly the individual sounds produced in the utterance of language. This has value beyond speech recognition as it can help identify the accent of the speaker and provides valuable insight to field linguists.

## Dataset

We used TIMIT, a corpus of spoken sentences annotated with phoneme information. It contains 630 English speakers with 8 distinct dialects, each reading 10 sentences. Table 1 contains the phonemes from the TIMIT dataset. Some phonemes bear great similarity, but each distinction is meaningful in the English language.

The large number of classes makes this task more comparable to CIFAR-100 than to MNIST. The baseline of most-common-value (the vowel /a/) achieves only 4\% accuracy on TIMIT, so we do not expect to see high accuracy scores. Rather, the question is how well we can improve on the baselines we establish.
<div align="center">
  
|      |      |      |      |      |      |      |      |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| a | ae | ah | ao | aw | adx | axr | ay |
| b | bcl | ch | d | dcl | dh | dx | eh |
| el | em | en | eng | epi | er | ey | f |
| g | gcl | h\# | hh | hv | ih | ix | iy |
| jh | k | kcl | l | m | n | ng | nx |
| ow | oy | p | pau | pcl | q | r | s |
| sh | t | tcl | th | uh | uw | ux | v |
| v | w | y | z | zh |

Table 1: All 61 phonemes in the TIMIT corpus, in their text representations
</div>

## Embeddings

Choosing an efficient representation of speech data is important, because segments of raw audio can lie in $\mathbb{R}^{1000}$ or more. For this reason it's important to be able to project the data into a lower-dimensional space. This map is referred to as an embedding. Embedding algorithms we used include resampling, the Mel spectrogram, random projections, and t-SNE. Each embedding was used in the preprocessing step for each model, and the results are reported in the Experiments section.

### Resampling

Phonemes vary in length. Vowels tend to be longer than consonants, and the shortest phonemes are often stops like /d/. (Figure 3 demonstrates the mean and variance of each phoneme in TIMIT.) Many of the models we use require input of fixed length, meaning that the raw audio must be resampled to the correct size in order to be passed as input to these models. One method of resampling involves interpolation of the raw audio samples, and then evaluation of the interpolating function to get the resampled points.

Short phonemes are often difficult to recognize because there are so few samples that the variance overpowers the signal. To provide better signal we pad every phoneme with up to 500 samples on either side. This increased the accuracy of our models by about 10\%. Unfortunately, padding also means that fewer samples actually come from the window where the phoneme is expressly being uttered; this could increase the model's confusion between phonemes that appear in similar contexts (e.g. /d/ and /t/ in "bat" and "bad").

### Mel spectrogram

The Fourier transform is a common signal processing technique that finds the representation of a segment of audio in terms of linear combinations of frequencies. An example of the difference between raw audio and the Fourier transform is shown in Figure 4. The melody filter (or mel filter) was developed to draw out the frequencies in a way that represents the range perceived by the human auditory system. The intuition of its use as an embedding is that this exact type of preprocessing occurs in the beginning stages of human auditory processing as well.

## Experiments: Classification

We start by producing simple baselines for classifying phonemes using common machine learning models. These achieve impressive performance given the 61 possible classes. We also train neural networks of two kinds, a residual network with 1-D convolutions and a recurrent neural network with LSTMs. We then compare results of these models on different embeddings.

### Random Forests

Random forests consistently lie among the top-performing classical machine learning methods. Random forests are an ensemble of decision trees, with each decision tree trained on a subset of the training features and a subset of the training data. We performed hyperparameter optimization over the number of decision trees included.

### KNN

K-nearest neighbors (KNN) is an approach that decides the classification of a new data point by taking the mode of the classes of its $k$ nearest neighbors in the data space. The accuracy of this model depends heavily on the choice of embedding. KNN has extremely low performance on raw audio, because audio segments that may be very similar but are shifted by half a wavelength will have a very large Euclidean distance. But over the right embedding space, KNN could provide a metric for an audio segment's "phoneme-ness", or likelihood of being a phoneme. This will be useful for the segmentation task in future work.

### RNN

Recurrent neural networks (RNN) allow neural networks to take input with variable size. We use a simple 1-D Convolution banks to downsize a sample of raw signal and run the variable-sized output through three layers of Long Short-Term Memory (LSTM) cells which summarize the output to a single vector. We then use two feed-forward layers to obtain a probability distribution of outputs. We achieved 73.0\% accuracy with this approach.

### 1-D CNNs

Convolutions have shown great success in speech and image classification, so we adopt the 1-D convolutional neural net (CNN) described by Rajpurkar et al. Their architecture is a deep CNN with 33 layers, residual connections, and an increasing number of channels combined with decreasing dimensionality. We found residual connections necessary to train anything deeper than 3 layers. Opting for a simpler architecture, we used only 9 layers and achieved 76.1\% top-1 accuracy and 92.3\% top-3 accuracy.
