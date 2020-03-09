import os
import glob
from scipy.io import wavfile
import pandas as pd
import numpy as np
import pickle
import get_preview as gp
import track_preparation as tp
import temp_audio_removal as tar
import vector_computing as vc
from keras.models import load_model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from python_speech_features import mfcc
from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for

app = Flask(__name__)


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

        # if not added that would be crap
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()

    return y_pred, fn_prob


def make_classification(stable_wav_filenames):
    # can remove if no classification, if we don't know true classes
    dictionary = {'fname':
                  stable_wav_filenames
                  }

    df = pd.DataFrame(data=dictionary)
    classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
               'Instrumental', 'International', 'Pop', 'Rock']  # all names of genres

    y_pred, fn_prob = build_predictions(
        'converted_track')  # predictions for all in this dir
    # acc_score = accuracy_score(y_true=y_true, y_pred=y_pred) #0.35687649164677804

    y_probs = []
    for i, row in df.iterrows():
        y_prob = fn_prob[row.fname]
        y_probs.append(y_prob)
        for c, p in zip(classes, y_prob):
            df.at[i, c] = p

    y_pred = [classes[np.argmax(y)] for y in y_probs]
    df['y_pred'] = y_pred
    df.to_csv('conv_results.csv', index=False)
    print('Saved df to csv successfully.')


# model preloading
model = load_model('models/conv.model')
p_path = os.path.join('pickles', 'convbig.p')
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)


@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    input_data = [{'artist': 'Worakls', 'track': 'Detached Motion', 'preview_url': 'https://p.scdn.co/mp3-preview/79ff02c11e68ec5c1ba5a4f6bb444afb167dcca0?cid=032bb2c730e645968318b1811d084943'}, {'artist': 'TOOL', 'track': 'Lateralus', 'preview_url': 'https://p.scdn.co/mp3-preview/4ad4447026761e5c467d92f46ae2f98e0b283699?cid=032bb2c730e645968318b1811d084943'}, {'artist': 'Worakls', 'track': 'Salzburg',
                                                                                                                                                                                                                                                                                                                                                                            'preview_url': 'https://p.scdn.co/mp3-preview/a8594e81712ca52e9f8ff308fe197fd47757cf06?cid=032bb2c730e645968318b1811d084943'}, {'artist': 'TOOL', 'track': 'Disposition', 'preview_url': 'https://p.scdn.co/mp3-preview/02b634cc83dc61d79bddcafef21d735e128800df?cid=032bb2c730e645968318b1811d084943'}, {'artist': 'Worakls', 'track': 'Nocturne', 'preview_url': 'https://p.scdn.co/mp3-preview/2136d9ae07c9f18ccdc9ac3a473c27c71568180b?cid=032bb2c730e645968318b1811d084943'}]
    print("---")
    gp.top_tracks_information(input_data)
    print("---")
    stable_wav_filenames = tp.to_wav()
    if stable_wav_filenames == []:
        print("no mp3 files, shutting down")
    else:
        for stable_wav_filename in stable_wav_filenames:
            tp.get_mfcc(stable_wav_filename)
        make_classification(stable_wav_filenames)
        tar.final_audio_cleaning()
        print('Cleaned up temporary audio files successfully')
        print(vc.cosine_distance_calculation())
        print('Predictions were made successfully')


if __name__ == "__main__":
    app.run(port=5001, debug=False)
