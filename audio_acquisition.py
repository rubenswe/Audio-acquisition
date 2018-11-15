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


def playrec_from_data(data, repetitions, delta_t, channels, samplerate=44100, chunk=8192):
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
    print(recorded_data.shape)
    #recorded_stereo_L = np.empty((len(data)+delta_t*samplerate, channels-1),dtype=np.float32)
    #recorded_stereo_R = np.empty((len(data)+delta_t*samplerate, channels-1),dtype=np.float32)

    #t = np.linspace(0, 1, 1024)

    p = pa.PyAudio()  # self? -> PlayRecorder()

    def callback(in_data, chunk, time_info, status):
        # in_data and out_data already given here. Have to assert which datatype is given?
        print(len(in_data), chunk)
        if channels == 1:
            in_data = np.fromstring(in_data, dtype=np.float32).reshape(chunk, channels)
            print(in_data.shape)
        elif channels == 2:
         #   
            #chunk_length = int(chunk/channels)  # chunk_stereo
            #print(chunk_length)
            in_data = np.fromstring(in_data, dtype=np.float32).reshape(chunk, channels)
            print(in_data.shape)
            #print(status)

        nonlocal recorded_data
        nonlocal i
        #nonlocal i_stereo
        #nonlocal recorded_stereo_L, recorded_stereo_R
        
        #if channels == 2:
         #   recorded_data[i:i+chunk, 0] = in_data[:, 0]
            #recorded_data[i:i+chunk, 1] = in_data[:, 1]

        # Last chunk of in_data might be shorter than chunk, so we zero-pad the left over samples
        if len(recorded_data[i:i+chunk]) != chunk:
            if channels == 1:
                recorded_data_pad = np.pad(recorded_data, (0, chunk-len(recorded_data[i:i+chunk])), "constant", constant_values=(0, 0))
                print("PADDED: ", len(recorded_data_pad))
                recorded_data_pad[i:i+chunk] = in_data
            elif channels == 2:
                print("PADDED STEREO")
                #recorded_data_pad[i_stereo:i_stereo+chunk_stereo, 0] = np.pad(recorded_data, (0, chunk-len(recorded_data[i_stereo:i_stereo+chunk_stereo, 0])), "constant", constant_values=(0, 0))
                #recorded_data_pad[i_stereo:i_stereo+chunk_stereo, 1] = np.pad(recorded_data, (0, chunk-len(recorded_data[i_stereo:i_stereo+chunk_stereo, 1])), "constant", constant_values=(0, 0))  # int(chunk-len())/2 ?
                
        else:
            if channels == 1:
                recorded_data[i:i+chunk] = in_data
            elif channels == 2:
                print("STEREO")
                #recorded_data[i:i+chunk] = in_data
                #print(in_data[:, 0].shape)
                #print(in_data[:, 1].shape)
                recorded_data[i:i+chunk, 0] = in_data[:, 0]  # Does the same as recorded_data[i:i+chunk] = in_data ?  # Left channel
                recorded_data[i:i+chunk, 1] = in_data[:, 1]  # Does the same as recorded_data[i:i+chunk] = in_data ?  # Right channel
                #recorded_data[i:i_stereo+chunk_stereo] = [in_data[offset::channels] for offset in range(channels)]
                #recorded_stereo_L[i_stereo:i_stereo+chunk_stereo, 0] = in_data[:chunk_stereo, 0]  # 1 dim
                #recorded_stereo_R[i_stereo:i_stereo+chunk_stereo, 0] = in_data[:chunk_stereo, 0]  # 1 dim
                #recorded_data = np.vstack((recorded_stereo_L, recorded_stereo_R)).T
                #print(recorded_data.shape)
            else:
                raise ValueError("Number of channels can only be 1 or 2")  # Will never go here, the stream raises an error when channels is not 1 or 2
        
        out_data = data_pad[i:i+chunk]
        print(len(data_pad[i:i+chunk]))

        i = i + chunk
        #i_stereo = i_stereo + chunk_stereo

        """if channels == 1:
            i += chunk
        elif channels == 2:
            i += int(chunk/2)"""
        print(status)
            
        # status: 
        # pa.paContinue = 0
        # pa.paComplete = 1
        # pa.paAbort = 2
        
        return (out_data, pa.paContinue)

    stream = p.open(format=pa.paFloat32,
                channels=channels,  # What to do in the case of 1 channel for playing and 2 for recording? Use 2 channels always?(leaving one channel empty in the mono case if possible)
                rate=samplerate,
                frames_per_buffer=chunk, # What should be used? Default: 1024
                input=True,
                output=True,
                stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.2)

    stream.stop_stream()
    stream.close()

    p.terminate()
    print(len(recorded_data))
    #print(status)

    return recorded_data


def playrec_stereo_rec():
    # This functionality should be incorporated in playrec() and playrec_from_data(). --> HOW?
    raise NotImplementedError
