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

## Experiments: Segmentation

### Reinforcement Learning

In a reinforcement learning environment, the problem is represented by
- an agent taking an action,
- a space of all possible actions that the agent can make,
- a space of possible states which the agent observes when making an action, and
- a reward the agent receives from making an action.

The goal is to train the agent to maximize the reward it can get by choosing actions that cause favorable states.

Consider a sliding window stretching out a certain distance from the beginning of an audio segment. We can represent the segmentation problem as a reinforcement learning environment by considering this window to be an agent that can either accept the current segment as a phoneme or expand the window to consider more samples of audio. We then define the state as the audio samples currently inside the window. Thus at each stage the window decides either to "expand" to consider more audio or "match" the current audio. The process then repeats starting at the end of the matched segment from the completed iteration.

For the reward function, we trained a machine learning model that is able to identify whether a given sample of audio is a phoneme segment or not. The architecture of this model is identical to the classification model, but instead of outputting a probability distribution, it outputs a single value that represents how likely it is that a given sample is a phoneme. We accomplished this task using the same TIMIT training data, and we generated a batch of random audio segments (not phonemes) from the same training audio.  

Let $U$ be the space of all possible audio segments of our training data and $P \subset U$ be the space of phoneme audio segments. $x \sim P$ is an audio segment from the training data and $z \sim U$ is a randomly generated audio segment. Let $R:U\rightarrow\mathbb R$ be a score function, mapping from the audio segment to a real number (high score means a likely phoneme, low score means not a phoneme).  We attempt to minimize the loss:

$`
\begin{align*}
L = &\mathbb E[\text{clip}\{R(z),-10,10\}] \\
 &- \mathbb E[\text{clip}\{R(x),-10,10\}] \\
 &- \mathbb E[|R(z)-\text{clip}\{R(z),-10,10\}|] \\
 &- \mathbb E[|R(x)-\text{clip}\{R(x),-10,10\}|]
\end{align*}
`$

where $`\text{clip}\{ a,b,c \} = \min\{\max\{a,b\},c\}`$.

Intuitively, minimizing the loss function will minimize the score we get from the randomly generated segments and maximize the score of the actual phoneme segments. We restrict the scores to be between -10 and 10 to avoid exploding scores, but we penalize the score for leaving this boundary. These penalties help the optimization avoid local minima. The explanation for each term in the loss is as follows.

- Minimize the output for random audio segments.
- Maximize the output for segments containing phonemes.
- Penalize scores outside $(-10,10)$ for random audio segments.
- Penalize scores outside $(-10,10)$ for segments containing phonemes.

Once we have a trained recognizer, we can use the scores it outputs as the reward for our reinforcement learning environment. We use Proximal Policy Optimization (PPO) algorithm to accomplish the reinforcement learning task. Since the states are segments of audio that are of variable length, we use the output of the LSTM so that the states are always a fixed size. The policy network and the value network of the PPO implementation are both composed of 5 fully connected layers. The policy network outputs a probability distribution of each action to take, and the agent samples an action from that distribution. This way, the agent can explore less probable actions while exploiting more probable actions. It then uses the output of the value network of the same input state in order to learn a better action distribution.

## Scoring

Since there isn't a standardized way of scoring a time series segmentation, we devised a distance metric for positively matched segments and also report the precision and recall. We start by pairing each predicted segment to the closest target segment. Some predicted segments will be paired with the same target segment, and so we remove the ones that are more distant. The unpaired predicted segments are counted as false positives while the unpaired target segments are counted as false negatives. Using these scoring methods, the reinforcement learning segmentation model positively matched segments within 9.398 ms on average, with precision of 70.65\% and recall of 88.62\%.
