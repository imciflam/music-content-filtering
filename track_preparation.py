import os
import glob
from pydub import AudioSegment
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank


def envelope(y, rate, threshold):  # signal envelope
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1,
                       center=True).mean()  # rolling window over data
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(y, rate):  # magnitute + freq (fft is complex)
    n = len(y)
    # amonth of time passing between each sample
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)  # fft is complex, division to balance
    return(Y, freq)


def to_wav():
    global standart_dir
    standart_dir = os.getcwd()
    audio_dir = os.path.join(os.getcwd(), 'current_track')
    extension = ('*.mp3')
    os.chdir(audio_dir)
    wav_filename = None
    for audio in glob.glob(extension):
        wav_filename = os.path.splitext(os.path.basename(audio))[0] + '.wav'
        AudioSegment.from_file(audio).export(
            standart_dir + "/converted_track/" + wav_filename, format='wav')
    os.chdir(standart_dir)
    if (wav_filename != None):
        return wav_filename
    else:
        print("wav_filename is null !!!")


def get_mfcc(wav_filename):
    dictionary = {'fname': [
        wav_filename,
    ]}

    df = pd.DataFrame(data=dictionary)
    df.set_index('fname', inplace=True)

    for f in df.index:
        if isinstance(f, str):
            rate, signal = wavfile.read(standart_dir + "/converted_track/"+f)
            df.at[f, 'length'] = signal.shape[0]/rate

    df.reset_index(inplace=True)

    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    wav_file = df.iloc[0, 0]  # condition -0 ind, 0 col
    signal, rate = librosa.load(
        standart_dir + "/converted_track/"+wav_file, sr=44100)  # sampling rate
    mask = envelope(signal, rate, 0.0005)  # can toggle 0.0005
    signal = signal[mask]  # remove noise
    signals[wav_file] = signal
    fft[wav_file] = calc_fft(signal, rate)
    # filterbank from pyspeechfeat, show a sec of data. nfft - calculated, short time Fourier trans
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[wav_file] = bank
    # calculated after discete cosine transform
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[wav_file] = mel
    # if more than one imported
    for f in tqdm(df.fname):
        signal, rate = librosa.load(
            standart_dir + "/converted_track/" + f, sr=16000)  # screw highfreq
        mask = envelope(signal, rate, 0.0005)  # clean up the junk
        wavfile.write(filename=standart_dir + "/converted_track/" +
                      f, rate=rate, data=signal[mask])
