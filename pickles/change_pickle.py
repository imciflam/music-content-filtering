import os
import pickle
import sys
sys.path.append('pickles')
sys.path.append('models')

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):  # filtered out
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        # 0.1 sec, how much data computing while creating window
        self.step = int(rate/10)
        self.model_path = os.path.join('models', 'convbig.model')
        self.p_path = os.path.join('pickles', 'convbig.p')

p_path = os.path.join('pickles', 'convbig.p')        
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    print(config)
