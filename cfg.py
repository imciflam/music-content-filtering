import os

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat = 13, nfft = 512, rate = 16000): #filtered out
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        self.step = int(rate/10) #0.1 sec, how much data computing while creating window
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
