from playrecorder import PlayRecorder, sf, pa, np, chirp
from audio_utilities import dbfft
import matplotlib.pyplot as plt 
from scipy.fftpack import ifft, fft, rfft, fftshift, fftfreq

"""
The main function is the application where the user interface with the PlayRecorder class
"""

def main():

    fs = 44100
    f1 = 20
    f2 = 20000
    T = 15
    t = np.linspace(0, T, T*fs)
    volume = 1
    w = (chirp(t, f1, T, f2, method='log', phi=90)*volume).astype(np.float32)

    pr_data = PlayRecorder(w)

    pr_data.recorded_data, pr_data.samplerate = sf.read("measurement1.wav")
    t_recording = np.linspace(0, pr_data.recorded_data.size, 16*fs) / fs
    plt.plot(t_recording, pr_data.recorded_data)
    plt.show()
    
    p = 20*np.log10(np.abs(np.fft.rfft(pr_data.recorded_data)))
    f = np.linspace(0, fs/2.0, len(p))
    print("REC: ", pr_data.recorded_data.size)
    print("samples: ", t_recording.size, "duration: ", pr_data.recorded_data.size / fs)
    print("DELAY: ", pr_data.get_delay(1), "seconds")

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(f, p)
    plt.xlim((40, fs/2))
    plt.title("Magnitude response")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Power(dB)")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t_recording, pr_data.impulse_response())
    plt.title('Impulse response')
    plt.xlabel("Time(s)")
    plt.ylabel("")
    plt.show()

    return 0

if __name__ == "__main__":
    main()