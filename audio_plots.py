import matplotlib.pyplot as plt 
import numpy as np 
from scipy.fftpack import ifft, fft, rfft, fftshift, fftfreq
import timeit
from audio_utilities import *

"""
Functions for plotting different properties of wave files.
    - Should data be an fft already? No need to compute it several times for each plot.(freq_data instead of data)
    - Make it part of an class?
"""

def plot_array(data):  # more parameters? like number of samples++
    pass


def plot_freq_resp(data, fs):
    start_time = timeit.default_timer()
    samples = data.shape[0] # Mono
    fftdata = fft(data)
    fftabs = abs(fftdata)
    freqs = fftfreq(samples, 1/fs)

    plt.xlim([10, fs/2])
    plt.xscale("log")
    plt.grid(True)
    plt.xlabel("Frequency [Hz]")
    plt.axvline(20, color='r')  # Start freq, param?
    plt.axvline(20000, color='r')  # End freq, param?

    plt.plot(freqs[:int(len(freqs)/2)], fftabs[:int(len(freqs)/2)])
    print(timeit.default_timer() - start_time)
    plt.show()


# Find delay, should be peak at delay, where in_data and out-data are equal
def plot_corr(in_data, out_data, fs, t):
    # Cross correlation is performed with numpy.correlate() with mode = 2.
    fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
    ax1.xcorr(in_data[:int(t*fs)], out_data[:int(t*fs)], usevlines=True, normed=True, lw=1)
    ax1.grid(True)
    ax1.axhline(0, color="black", lw=1)
    ax1.set_title("Crosscorrelation between input and output")
    # ax1.axvline(get_latency(), color="red", lw=1)

    ax2.acorr(out_data[:int(t*fs)], usevlines=True, normed=True, maxlags=None, lw=1)
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title("Autocorrelation of output")

    plt.show()


def plot_corr2(in_data, out_data, fs, t):
    corr = np.correlate(in_data[:int(t*fs)], out_data[:int(t*fs)], mode="same")
    plt.plot(corr)
    plt.show()


def plot_power_spectrum(data):
    pass


def plot_magnitude_response(tf, fs):
    #samples = len(tf)
    #freqs = fftfreq(samples, 1/fs)

    plt.plot(20*np.log(np.abs(tf)))  # log10?
    plt.show()


def plot_transfer_function(in_data, out_data, fs):  # plot_magn_and phase?
    L = len(out_data)
    in_pad = pad_input_to_output_length(in_data, out_data)
    in_fft = fft(in_pad) / L
    out_fft = fft(out_data) / L
    # t = np.linspace(0, L, L) / fs
    tf = out_fft / in_fft
    fn = fs/2  # ?
    fv = np.linspace(0, 1, np.floor(L/2)+1)*fn

    fig, ax = plt.subplots()
    #ax.semilogy(fv, np.abs(tf[:len(fv)])*2)
    #plt.plot(fv, 20*np.log10(np.abs(tf[:len(fv)])))
    #ax.semilogy(fv, 20*np.log10(np.abs(tf[:len(fv)])))
    ax.grid()
    #plt.plot(t, in_pad, t, out_data)
    #plt.plot(fv, np.imag(tf[:len(fv)]))
    #plt.yscale("symlog")
    #plt.plot(np.angle(tf))

    spectrum, freqs, line = plt.phase_spectrum(out_data, fs)
    #spectrum, freqs, line = plt.magnitude_spectrum(tf, fs)
    #pxx, freqs = plt.psd(tf, NFFT=None, Fs=fs)  # in_data, out_data
    #spectrum, freqs, line = plt.angle_spectrum(tf, fs)

    plt.plot(freqs, spectrum)
    plt.show()