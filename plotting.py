import keras.models
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


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
    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
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
            y_true.append(c)

        # or that would be crap
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()

    return y_true, y_pred, fn_prob


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
               'Instrumental', 'International', 'Pop', 'Rock']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# can remove if no classification, if we don't know true classes
df = pd.read_csv('new_try.csv')
classes = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
           'Instrumental', 'International', 'Pop', 'Rock']

fn2class = dict(zip(df.fname, df.label))
p_path = os.path.join('pickles', 'convbig.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model('models/conv.model')

y_true, y_pred, fn_prob = build_predictions(
    'valrandmusic')  # predictions for all in this dir

print('Confusion Matrix')
results = confusion_matrix(y_true=y_true, y_pred=y_pred)
print(results)

print('Accuracy Score:')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)
print(acc_score)

print('Report')
cl = classification_report(y_true=y_true, y_pred=y_pred)
print(cl)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
