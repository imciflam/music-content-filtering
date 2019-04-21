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



def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []

        for i in range(0, wav.shape[0]-config.step, config.step):#cannot go further
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep = config.nfeat, nfilt = config.nfilt, nfft = config.nfft)
            x = (x - config.min)/(config.max - config.min)#range
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode=='time':
                x = np.expand_dims(x, axis = 0) #expand to 1 sample
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)

        fn_prob[fn] = np.mean(y_prob, axis = 0).flatten() #or that would be crap

    return y_true, y_pred, fn_prob

df = pd.read_csv('instruments200.csv')#can remove if no classification
classes = list(np.unique(df.label))#all names of genres 

fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'conv.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean200')
acc_score = accuracy_score(y_true = y_true, y_pred = y_pred)

y_probs = []
for i,row in df.iterrows():
    print(row.fname)
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c ,p in zip(classes, y_prob):
        df.at[i, c] = p

print(classes)
y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions200.csv', index = False)


