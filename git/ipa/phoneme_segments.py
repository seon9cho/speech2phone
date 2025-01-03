# Collect samples of the phonemes into their respective folders in the current directory.
# Kyle Roth and Seong-Eun. 2018-12-12.

from scipy.io import wavfile

index_dict = {
    's': [
        ('16k-wav/pulmonic_consonant/Voiceless_alveolar_sibilant.wav', 485, 6792),
        ('16k-wav/pulmonic_consonant/Voiceless_alveolar_sibilant.wav', 19372, 24778),
        ('../sample/00001.wav', 31621, 34222),
        ('../sample/00001.wav', 75044, 78526),
        ('../sample/00001.wav', 107333, 109911),
        ('../sample/00001.wav', 143333, 146576),
        ('../sample/00002.wav', 24697, 27824),
        ('../sample/sample0.wav', 24922, 27376),
        ('../sample/sample0.wav', 103594, 105711),
        ('../sample/sample0.wav', 126243, 128675),
        ],
    'f': [
        ('16k-wav/pulmonic_consonant/Voiceless_labiodental_fricative.wav', 3548, 8210), 
        ('../sample/sample2.wav', 25287, 26881),
        ('../sample/sample2.wav', 36475, 37463),
        ('../sample/sample7.wav', 83983, 85360),
        ('../sample/sample0.wav', 122851, 124376),
        ('../sample/sample1.wav', 40545, 42269),
        ('../sample/sample1.wav', 102203, 103402),
        ('../sample/sample1.wav', 105021, 107187),
        ('../sample/sample3.wav', 37205, 38424),
        ('../sample/sample3.wav', 43591, 45277)
        ],
    'g': [
        ('16k-wav/pulmonic_consonant/Voiced_velar_plosive_02.wav', 3153, 3978),
        ('16k-wav/pulmonic_consonant/Voiced_velar_plosive_02.wav', 20314, 20903),
        ('../sample/sample7.wav', 46249, 47172),
        ('../sample/sample0.wav', 81244, 82293),
        ('../sample/sample3.wav', 93316, 94895),
        ('../sample/sample7.wav', 46233, 47109),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1523929, 1524953),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2027652, 2028471),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 3050704, 3051709),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 3614864, 3615891),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 5418706, 5420204)
        ],
    'p': [
        # ('16k-wav/pulmonic_consonant/Voiceless_bilabial_plosive.wav', 0, 1342),       # these are unaspirated
        # ('16k-wav/pulmonic_consonant/Voiceless_bilabial_plosive.wav', 15881, 17296),  # these are unaspirated
        ('../sample/00001.wav', 28816, 30193),
        ('../sample/00001.wav', 104679, 106107),
        ('../sample/sample1.wav', 122284, 125069),
        ('../sample/sample4.wav', 20015, 21620),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1290868, 1291719),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1918297, 1919053),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2350519, 2351590),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2445208, 2445980),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2567417, 2568331),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2588636, 2589770)
        ],
    't': [
        ('../sample/sample0.wav', 68104, 69578),
        ('../sample/sample0.wav', 108478, 109902),
        ('../sample/sample4.wav', 77851, 79478),
        ('../sample/sample6.wav', 7802, 9000),
        ('../sample/sample6.wav', 21993, 23652),
        ('../sample/sample8.wav', 31614, 33369),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1386329, 1387905),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1498803, 1500316),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1627361, 1628621),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1647650, 1648784)
        ],
    'schwa': [
        ('../sample/00001.wav', 6087, 8187),
        ('../sample/00001.wav', 33596, 35200),
        ('../sample/00001.wav', 112245, 114452),
        ('../sample/sample0.wav', 131931, 134065),
        ('../sample/sample1.wav', 50328, 52498),
        ('../sample/sample1.wav', 113092, 113723),
        ('../sample/sample1.wav', 23121, 25200),
        ('../sample/sample3.wav', 27780, 29575),
        ('../sample/sample3.wav', 89773, 91296),
        ('../sample/sample3.wav', 92667, 93716)
        ],
    'a': [
        ('../sample/sample3.wav', 45297, 46906),
        ('../sample/sample3.wav', 76589, 78670),
        ('../sample/sample4.wav', 51492, 54878),
        ('../sample/sample4.wav', 52670, 54724),
        ('../sample/sample4.wav', 24799, 26539),
        ('../sample/sample4.wav', 101849, 104171),
        ],
    'ae': [
        ('16k-wav/vowel/Near-open_front_unrounded_vowel.wav', 344, 6509),
        ('../sample/sample0.wav', 82477, 84091),
        ('../sample/sample5.wav', 32536, 33463),
        ('../sample/sample8.wav', 40745, 41896),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1501009, 1502080),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1525882, 1527457),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1539366, 1539981),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1724476, 1725358),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2446799, 2447791),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2698242, 2699811)
        ],
    'i': [
        ('16k-wav/vowel/Close_front_unrounded_vowel.wav', 287, 7102),
        ('../sample/sample1.wav', 13879, 15856),
        ('../sample/sample1.wav', 92740, 93897),
        ('../sample/sample4.wav', 28741, 30232),
        ('../sample/sample4.wav', 81817, 82495),
        ('../sample/sample5.wav', 72715, 73410),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1584466, 1585916),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 1883326, 1883988),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2317658, 2318478),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2418428, 2419106),
        ('../sample/2018-10-1010-russell-m-nelson-64k-eng.wav', 2844196, 2845078)
        ]
}


def collect():
    """Extract sound bytes containing just the phonemes.
    
    For now, the indices are hard-coded in `index_dict`.
    """
    for char in index_dict.keys():
        i = 0
        for path, start, end in index_dict[char]:
            # read in the audio
            exp_path = char + '/' + char + str(i) + '.wav'
            rate, samples = wavfile.read(path)
            wavfile.write(exp_path, rate, samples[start:end])
            i+=1



