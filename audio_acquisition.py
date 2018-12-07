import soundfile as sf 
import pyaudio as pa 
import numpy as np 
import time
from audio_utilities import filename_generator
import matplotlib.pyplot as plt

"""
    Functions for playing and recording wave files.
"""

def playrec(in_file, out_file):  # take in numpy arrays of soundfile objects as parameters?
    """
    Function for playing a test sound(mono) and simultaneously recording(mono/stereo) the response(wire_callback.py) 
    """
    # Duration given by in_file, repeats only in test case? --> tests.py
    # Should return data of record file?

    input_file = sf.SoundFile(in_file, "rb")  # file to be played
    output_file = sf.SoundFile(out_file, "wb", input_file.samplerate, input_file.channels, input_file.subtype)  # File with the recorded data

    p = pa.PyAudio()

    # in_data is the recorded data. callback() is called in a separate thread
    def callback(in_data, frame_count, time_info, status):
        in_data = np.fromstring(in_data, dtype=np.float32).reshape(frame_count, input_file.channels)  # Convert the data before writing
        out_data = input_file.read(frame_count, dtype=np.float32)  # Read in the data to be played(and then recorded into in_data)
        output_file.write(in_data)  # Write one frame of the recorded data to the output file, NOT recommended to do inside callback()!! Alternative way?
        return (out_data, pa.paContinue)  # out_data is the played data

    # Could store in_data in a buffer inside callback() and write to file later?

    stream = p.open(format=pa.paFloat32,
                channels=input_file.channels,  # What to do in the case of 1 channel for playing and 2 for recording? Use 2 channels always?(leaving one channel empty in the mono case if possible)
                rate=input_file.samplerate,
                input=True,
                output=True,
                stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.2)

    stream.stop_stream()
    stream.close()

    p.terminate()

    input_file.close()
    output_file.close()


def playrec_from_data(data, repetitions, delta_t, channels, samplerate=44100, chunk=1024):
    """
    Plays data and records it. Can be repeated and record for delta_t longer than it is playing.
    """
    # What to do with channels here ? In the case of mono in(playback) and stereo out(recording) ?
    # return data or file? both?
    # Better name for data?
    # Relation between data and channels? Played/recorded

    assert type(data) == np.ndarray, "Given data is not an numpy array"
    i = 0

    data_pad = np.pad(data, (0, samplerate*delta_t), "constant", constant_values=(0, 0))
    recorded_data = np.empty((len(data)+delta_t*samplerate, channels), dtype=np.float32)  # What about for stero recording? --> One extra dimension divide len by channels?
    #print(recorded_data.shape)
    #recorded_stereo_L = np.empty((len(data)+delta_t*samplerate, channels-1),dtype=np.float32)
    #recorded_stereo_R = np.empty((len(data)+delta_t*samplerate, channels-1),dtype=np.float32)

    #if channels == 2:
     #   chunk = int(chunk / 2)

    p = pa.PyAudio()  # self? -> PlayRecorder()

    def callback(in_data, chunk, time_info, status):
        # in_data and out_data already given here. Have to assert which datatype is given?

        nonlocal recorded_data
        nonlocal i

        if not in_data:
            print("No input")

        print(len(in_data), chunk)
        if channels == 1:
            in_data = np.fromstring(in_data, dtype=np.float32).reshape(chunk, channels)
        elif channels == 2:
            #chunk = int(chunk / 2)
            in_data = np.fromstring(in_data, dtype=np.float32)
            in_data = np.reshape(in_data, (int(len(in_data)/channels), channels))
            #in_data_L = np.fromstring(in_data[0::2], dtype=np.float32)
            #in_data_R = np.fromstring(in_data[1::2], dtype=np.float32)
            #ch1 = in_data[0::2, 0]
            #ch2 = in_data[1::2, 1]
            #print("ch1: ", ch1.shape)
            #print("ch2: ", ch2.shape)
      


        # Last chunk of in_data might be shorter than chunk, so we zero-pad the left over samples
        print("DIFF: ", chunk, "-", len(recorded_data[i:i+chunk]), "=", chunk-len(recorded_data[i:i+chunk]))
        if len(recorded_data[i:i+chunk]) < chunk:
            if channels == 1:
                recorded_data_pad = np.pad(recorded_data[:, 0], (0, chunk-len(recorded_data[i:i+chunk])), "constant", constant_values=(0, 0))
                print("DIFF: ", chunk, "-", len(recorded_data[i:i+chunk]), "=", chunk-len(recorded_data[i:i+chunk]))
                print("RECDATA DIFF: ", len(recorded_data_pad), "-", len(recorded_data), "=", len(recorded_data_pad)-len(recorded_data))
                #plt.plot(recorded_data)
                #plt.show()
                print("IN DATA: ", in_data.shape, len(in_data), len(in_data[:, 0]))
                print("RECDATA PADDED: ", recorded_data_pad.shape, len(recorded_data_pad[i:i+chunk]), len(recorded_data_pad[i:i+(chunk-len(recorded_data[i:i+chunk]))]))
                #recorded_data_pad[i:i+chunk] = in_data[:, 0]
                recorded_data_pad[i:i+chunk] = in_data[:len(recorded_data_pad[i:i+chunk]), 0]
                return (None, pa.paComplete)
                #recorded_data_pad[i:i+(chunk-len(recorded_data[i:i+chunk]))] = in_data[:, 0]
                # len(recorded_data) % chunk = x -----> x is the length of the last chunk ?
            elif channels == 2:
                print("PADDED STEREO")
                #plt.plot(recorded_data)
                #plt.show()
                print(recorded_data.shape)
                #recorded_data_pad = n...
                recorded_data_pad = np.pad(recorded_data, (0, chunk-len(recorded_data[i:i+chunk])), "constant", constant_values=(0, 0))
                print("DIFF: ", chunk, "-", len(recorded_data[i:i+chunk]), "=", chunk-len(recorded_data[i:i+chunk]))
                print(recorded_data_pad.shape)
                recorded_data_pad[i:i+chunk, 0] = in_data[:len(recorded_data_pad[i:i+chunk]), 0]
                recorded_data_pad[i:i+chunk, 1] = in_data[:len(recorded_data_pad[i:i+chunk]), 1]
                return (None, pa.paComplete)
                #recorded_data_pad[i:i+chunk, 0] = np.pad(recorded_data, (0, chunk-len(recorded_data[i:i+chunk, 0])), "constant", constant_values=(0, 0))
                #recorded_data_pad[i:i+chunk, 1] = np.pad(recorded_data, (0, chunk-len(recorded_data[i:i+chunk, 1])), "constant", constant_values=(0, 0))  # int(chunk-len())/2 ?
                
        else:
            if channels == 1:
                recorded_data[i:i+chunk] = in_data
            elif channels == 2:
                print("STEREO")
                print(recorded_data.shape)
                #recorded_data[i:i+chunk] = in_data
                #print(in_data[:, 0].shape)
                #print(in_data[:, 1].shape)
                recorded_data[i:i+chunk, 0] = in_data[:, 0]  # Does the same as recorded_data[i:i+chunk] = in_data ?  # Left channel
                recorded_data[i:i+chunk, 1] = in_data[:, 1]  # Does the same as recorded_data[i:i+chunk] = in_data ?  # Right channel
                #recorded_data[i:i+chunk, 0] = in_data_L
                #recorded_data[i:i+chunk, 1] = in_data_R
                #plt.plot(recorded_data)
                #plt.show()
                #recorded_data[i:i+int(chunk/2), 0] = ch1
                #recorded_data[i:i+int(chunk/2), 1] = ch2
                #recorded_data[i:i_stereo+chunk_stereo] = [in_data[offset::channels] for offset in range(channels)]
                #recorded_stereo_L[i_stereo:i_stereo+chunk_stereo, 0] = in_data[:chunk_stereo, 0]  # 1 dim
                #recorded_stereo_R[i_stereo:i_stereo+chunk_stereo, 0] = in_data[:chunk_stereo, 0]  # 1 dim
                #recorded_data = np.vstack((recorded_stereo_L, recorded_stereo_R)).T
                #print(recorded_data.shape)
            else:
                raise ValueError("Number of channels can only be 1 or 2")  # Will never go here, the stream raises an error when channels is not 1 or 2
        
        out_data = data_pad[i:i+chunk]  #-----> WORKS FOR CHANNELS == 1, NOT CHANNELS == 2
        #out_data = data_pad  # TOO SLOW? BAD PLOT FOR BOTH CHANNELS = 1, 2 .... 
        print("OUT:", data_pad.shape)
        print(len(data_pad[i:i+chunk]))

        # Traverse one chunk at a time
        print("INDEX: ", i, i+chunk)
        i = i + chunk
        #i_stereo = i_stereo + chunk_stereo
        #if i == 100*chunk:
           # plt.plot(recorded_data)
            #plt.show()
            
        # status: 
        # pa.paContinue = 0
        # pa.paComplete = 1
        # pa.paAbort = 2
        print("STATUS: ", status)
        print("STREAM: ", stream.is_active())
        return (out_data, pa.paContinue)

    stream = p.open(format=pa.paFloat32,
                channels=channels,  # What to do in the case of 1 channel for playing and 2 for recording? Use 2 channels always?(leaving one channel empty in the mono case if possible)
                rate=samplerate,
                frames_per_buffer=chunk, # What should be used? Default: 1024
                input=True,
                output=True,
                stream_callback=callback)

    stream.start_stream()  # callback() is called here
    
    plt.plot(data_pad)
    plt.show()
    #print("STREAM: ", stream.is_active())

    while stream.is_active():
        time.sleep(0.2)

    stream.stop_stream()
    stream.close()

    p.terminate()
    print(len(recorded_data))
    #print(status)

    return recorded_data


def playrec_stereo_rec(data, repetitions, delta_t, channels, samplerate=44100, chunk=1024):
    assert type(data) == np.ndarray, "Given data is not an numpy array"
    i = 0

    #data_pad = np.pad(data, (0, samplerate*delta_t), "constant", constant_values=(0, 0))
    recorded_data = np.empty((len(data)+delta_t*samplerate, channels), dtype=np.float32)  
    #out_data = data
    p = pa.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        nonlocal recorded_data
        nonlocal i
    
        in_data = np.fromstring(in_data, dtype=np.float32)
        print("INDATA: ", in_data.shape)

        #if int(len(in_data[0::2])) != len(recorded_data[i:i+int(frame_count):2, 0]):
            # len(recorded_data) % chunk = x -----> x is the length of the last chunk ?
            #print("FINISHED.")
            #return (data, pa.paComplete)

        #recorded_data[i:i+int(frame_count*2):2, 0] = in_data[0::2]
        #recorded_data[i+1:i+int(frame_count*2):2, 1] = in_data[1::2]
        recorded_data = in_data
        print("INDEX: ", i, i+frame_count)
        i += frame_count

        if i == len(recorded_data):
            return (data, pa.paComplete)

        return (data, pa.paContinue)

    stream = p.open(format=pa.paFloat32,
                channels=channels,  # What to do in the case of 1 channel for playing and 2 for recording? Use 2 channels always?(leaving one channel empty in the mono case if possible)
                rate=samplerate,
                frames_per_buffer=4096, # What should be used? Default: 1024, 0 => automatic/variable
                input=True,
                output=True,
                stream_callback=callback)

    stream.start_stream()  # callback() is called here

    while stream.is_active():
        time.sleep(0.2)

    stream.stop_stream()
    stream.close()

    p.terminate()

    plt.plot(recorded_data)
    plt.show()

    return 0



def callback_example(in_data, frame_count, time_info, flag):
    global data, recording, ch1, ch2 
    data = np.fromstring(in_data, dtype=np.float32) 
    ch1=data[0::2] 
    ch2=data[1::2] 
    return (in_data, recording)
