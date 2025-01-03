from scipy.io import wavfile

start_index = 2123401
end_index = 2139048
window_size = 2500
stride = 10
f_name = "../../sample/2018-10-1010-russell-m-nelson-64k-eng.wav"

def collect():
    """Extract sound bytes containing just the phonemes.
    
    For now, the indices are hard-coded in `index_dict`.
    """
    rate, samples = wavfile.read(f_name)
    start = start_index
    end = start_index + window_size
    while end < end_index:
    	exp_path = 'audio/' + str(start) + '-' + str(end) + '.wav'
    	wavfile.write(exp_path, rate, samples[start:end])
    	start += stride
    	end += stride

collect()