from playrecorder import PlayRecorder, sf, pa, np, chirp
import matplotlib.pyplot as plt 
#import playrecorder

"""
The main function is the application where the user interface with the PlayRecorder class(?)
"""

def main():

    fs = 44100
    f1 = 20
    f2 = 2000
    T = 5
    t = np.linspace(0, T, T*fs)
    volume = 0.5
    w = (chirp(t, f1, T, f2, method='linear', phi=90)*volume).astype(np.float32)

    pr_data = PlayRecorder(w)
    #pr_data.playrec_from_data(1)  # delta_t has to be > 0 to be able to record the last samples?

    #plt.plot(pr_data.recorded_data)
    #plt.plot(pr_data.recorded_data_padded)
    #plt.show()
    
    return 0


if __name__ == "__main__":
    main()