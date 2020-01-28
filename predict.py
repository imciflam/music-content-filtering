import keras.models
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from python_speech_features import mfcc
import pickle


class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):  # filtered out
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        # 0.1 sec, how much data computing while creating window
        self.step = int(rate/10)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', 'convbig.p')


def build_predictions(audio_dir):
    # y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        y_prob = []

        for i in range(0, wav.shape[0]-config.step, config.step):  # cannot go further
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat,
                     nfilt=config.nfilt, nfft=config.nfft)
            x = (x - config.min)/(config.max - config.min)  # range
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x, axis=0)  # expand to 1 sample
            elif config.mode == 'convtime':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            # y_true.append(c)

        # if not added that would be crap
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()

    return y_pred, fn_prob


# can remove if no classification, if we don't know true classes
dictionary = {'fname': ['algo.wav',
                        'badguys.wav',
                        'believer.wav',
                        'chopinmajs.wav',
                        'chopins.wav',
                        'disturbed.wav',
                        'dove.wav',
                        'experiences.wav',
                        'gomez.wav',
                        'helas.wav',
                        'iawfyls.wav',
                        'kazak.wav',
                        'luna.wav', 
                        ]}

df = pd.DataFrame(data=dictionary)
#df = pd.read_csv('new_try.csv')
classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
           'Instrumental', 'International', 'Pop', 'Rock']  # all names of genres

# fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'convbig.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model('models/conv.model')

y_pred, fn_prob = build_predictions(
    'valrandmusic')  # predictions for all in this dir
# acc_score = accuracy_score(y_true=y_true, y_pred=y_pred) #0.35687649164677804

y_probs = []
# for i in list:
#    print(i)
for i, row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i, c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('conv_results.csv', index=False)
