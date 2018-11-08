import soundfile as sf 
import numpy as np
from scipy.fftpack import ifft, fft, rfft, fftshift, fftfreq
import scipy.signal as sig 
import matplotlib.pyplot as plt


def filter20_to_20k(x, fs):
    nyq = 0.5 * fs
    sos = sig.butter(5, [20.0 / nyq, 20000.0 / nyq], btype="band", output="sos")
    return sig.sosfilt(sos, x)


def normalize(data):
    assert isinstance(data, np.ndarray), "Assertion failed, input data is not an numpy array."
    return (data - data.min()) / (np.ptp(data))


def pad_input_to_output_length(x, y):  # Use to find transfer function?
    """
    x: input array
    y: output array
    """
    # Output is always longer or the same length as input because of latency
    assert len(x) != len(y), "Arrays are already of the same length"
    x_pad = np.pad(x, (0, len(y)-len(x)), "constant", constant_values=(0, 0))
    return x_pad

def impulse_response(x, fs):
    # Inverse filter:
    T = x.shape[0] / fs  # Or take as parameter? 
    t = np.arange(0,T*fs-1) / fs
    R = np.log(20/2000)  # Ratio? What is this for? sweep is made in audio_utilites_test.py
    k = np.exp(t*R/T)  # How to access this information from the input? --> Take in SoundFile object or NumPy array? --> Just take in as parameters?
    f = x[::-1] / k
    return sig.fftconvolve(x, f, mode="same")


def transfer_function(x, y):
    x_pad = pad_input_to_output_length(x, y)
    x_fft = rfft(x_pad)  / len(y)
    y_fft = rfft(y)  / len(y)
    return y_fft / x_fft  # Arrays have to be the same length => Zero-pad with pad_input_to_output_length()


def get_delay(in_data, out_data, fs, t):
    # ax1.xcorr: Cross correlation is performed with numpy.correlate() with mode = 2("same").
    corr = np.correlate(in_data[:int(t*fs)], out_data[:int(t*fs)], mode="same")  # mode="same" makes it very slow => use small t
    return (int(len(corr)/2) - np.argmax(corr)) / fs  # seconds


def filename_generator(filename, N):
    filenames = []
    for i in range(N):
        name = filename + str(i) + ".wav"
        filenames.append(name)
    return filenames


def get_available_devices():
    # Implemented in class PlayRecorder()
    raise NotImplementedError 


def generate_sine():
    pass


def generate_sweep():
    pass