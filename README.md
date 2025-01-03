# *speech2phone*: Phoneme Segmentation and Classification from Raw Audio

*We extend the speech recognition problem by converting raw audio data into discrete phonemes, rather than directly into language. For classifying phonemes, we combine several classical machine learning techniques such as dimensionality reduction with the expressive power of deep neural nets. We compare recurrent and convolutional neural networks and various embeddings, achieving 74.9\% on the TIMIT dataset. We also attempt phoneme segmentation, getting an average error of 9.398 ms with precision 70.65\% and recall 88.62\% using a reinforcement learning model.*

## Introduction

Speech recognition is a fundamental discipline of computational linguistics. It has traditionally been a very difficult problem to solve even with statistical models but has recently seen marked improvement through the use of neural networks.

Similar to speech recognition, phoneme recognition is the challenge of transcribing audio input into a phonetic alphabet, e.g. the International Phonetic Alphabet (IPA). While a language's regular alphabet and spelling are decided by history and don't necessarily match pronunciation, the goal of a phonetic alphabet is to represent exactly the individual sounds produced in the utterance of language. This has value beyond speech recognition as it can help identify the accent of the speaker and provides valuable insight to field linguists.

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

<p style="text-align:center;">Text_content</p>
<p style="text-align:center;">Sample text with center alignment</p>
