import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def envelope(y, rate, threshold): #signal envelope
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate/10), min_periods = 1, center = True).mean() #rolling window over data
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
    
def calc_fft(y,rate): #magnitute + freq (fft is complex)
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate) #amonth of time passing between each sample
    Y = abs(np.fft.rfft(y)/n)#fft is complex, division to balance
    return(Y, freq)

df = pd.read_csv("instruments3.csv") #import excel
df.set_index('fname', inplace = True)

for f in df.index:
    rate, signal = wavfile.read('wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()#middle length of each one

fig, ax = plt.subplots()
ax.set_title('Class distr', y=1.00)
ax.pie(class_dist, labels = class_dist.index, autopct='%1.1f%%', shadow = False, startangle = 90)  
ax.axis('equal') #for circle
plt.show()
df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.label ==c].iloc[0,0] #condition -0 ind, 0 col
    signal, rate = librosa.load('wavfiles/'+wav_file, sr = 44100) #sampling rate
    mask = envelope(signal, rate, 0.0005) #can toggle 0.0005
    signal = signal[mask] #remove noise
    signals[c]=signal
    fft[c] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T #filterbank from pyspeechfeat, show a sec of data. nfft - calculated, short time Fourier trans
    fbank[c]=bank
    mel = mfcc(signal[:rate],rate,numcep=13, nfilt =26, nfft = 1103).T#calculated after discete cosine transform
    mfccs[c] = mel

#plot_signals(signals) #remove dead spaces
#plt.show()

#plot_fft(fft) #fourier
#plt.show()

#plot_fbank(fbank) #filterbank energies
#plt.show()

#plot_mfccs(mfccs) 
#plt.show()

if len(os.listdir('clean'))==0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('wavfiles/' + f, sr = 16000) #screw highfreq
        mask = envelope(signal, rate, 0.0005) #clean up the junk 
        wavfile.write(filename='clean/'+f, rate = rate, data= signal[mask])
