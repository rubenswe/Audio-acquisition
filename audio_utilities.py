import soundfile as sf 
import numpy as np
from scipy.fftpack import ifft, fft, rfft
import scipy.signal as sig 


def plot_correlation(parameter_list):
    pass


def filter20_to_20k(x, fs):
    nyq = 0.5 * fs
    sos = sig.butter(5, [20.0 / nyq, 20000.0 / nyq], btype="band", output="sos")
    return sig.sosfilt(sos, x)


def plot_spectrum(parameter_list):
    pass

def normalize(data):
    assert isinstance(data, np.ndarray), "Assertion failed, input data is not an numpy array."
    return (data - data.min()) / (np.ptp(data))


def pad_input_to_output_length(x, y):
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


def generate_sine():
    pass


def generate_sweep():
    pass