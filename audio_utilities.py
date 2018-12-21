import soundfile as sf 
import numpy as np
from scipy.fftpack import ifft, fft, rfft, fftshift, fftfreq
import scipy.signal as sig 
from scipy.signal import chirp
import matplotlib.pyplot as plt


def filter20_to_20k(x, fs):
    nyq = 0.5 * fs
    sos = sig.butter(5, [20.0 / nyq, 20000.0 / nyq], btype="band", output="sos")
    return sig.sosfilt(sos, x)


def normalize(data):
    assert isinstance(data, np.ndarray), "Assertion failed, input data is not an numpy array."
    return (data - data.min()) / (np.ptp(data))


def pad_input_to_output_length(x, y):
    """
    x: input array
    y: output array
    """
    # Output is always longer or the same length as input because of delay
    assert len(x) != len(y), "Arrays are already of the same length"
    x_pad = np.pad(x, (0, len(y)-len(x)), "constant", constant_values=(0, 0))
    return x_pad


def get_delay(in_data, out_data, fs, t):
    corr = np.correlate(in_data[:int(t*fs)], out_data[:int(t*fs)], mode="same")  # mode="same" makes it very slow => use small t
    return (int(len(corr)/2) - np.argmax(corr)) / fs  # seconds


def filename_generator(filename, N):
    filenames = []
    for i in range(N):
        name = filename + str(i) + ".wav"
        filenames.append(name)
    return filenames


def get_available_devices():
    raise NotImplementedError 


def decode(in_data, channels):
    in_data = np.fromstring(in_data, dtype=np.float32)
    # ...
    raise NotImplementedError


def encode(data):
    raise NotImplementedError


def generate_sweep(f1, f2, fs, T, volume):
    assert volume > 0 and volume <= 1, "The volume must be between 0 and 1"
    t = np.linspace(0, T, T*fs)
    w = (chirp(t, f1, T, f2, method='linear', phi=90)*volume).astype(np.float32)
    return w


def round_up_to_multiple(number, multiple):
    num = number + (multiple - 1)
    return num - (num % multiple)