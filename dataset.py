import os
import math
import numpy
import torch
import librosa
from torch.utils.data.dataset import Dataset

import modules

class AudioData(Dataset):
    def __init__(self, seq, resample, batch):
        self.data = []
        self.seq_len = seq+1
        self.min_signal_len = 10**10

        # all tracks
        #for dir in tqdm.tqdm(os.listdir(os.sep.join([r".\archive", "genres"]))):
        #    for wav in os.listdir(os.sep.join([r".\archive", "genres", dir])):
        #        #audio loading w/o resample
        #        signal,_ = librosa.load(os.sep.join([r".\archive", "genres", dir, wav]), sr=RESAMPLE)
        #        self.data.append(signal)
        #
        #        if len(signal) < self.min_signal_len:
        #            self.min_signal_len = len(signal)

        # blues
        #for wav in os.listdir(os.sep.join([r".\archive", "genres", "blues"])):
        #    signal,_ = librosa.load(os.sep.join([r".\archive", "genres", "blues", wav]), sr=RESAMPLE)
        #    self.data.append(signal)

        #    if len(signal) < self.min_signal_len:
        #        self.min_signal_len = len(signal)

        # only blues 00000
        self.data.append(librosa.load(os.sep.join([r".\archive", "genres", "blues", "blues.00000.wav"]), sr=resample)[0])
        self.min_signal_len = len(self.data[0])

        self.data = numpy.array(self.data)

        datapoints = len(self.data)*(self.min_signal_len-self.seq_len)
        self.batches = math.ceil(datapoints / batch)

        print("data points:", datapoints)
        print("possible batches:", self.batches)

    def __len__(self):
        return len(self.data)*(self.min_signal_len-self.seq_len)

    def __getitem__(self, idx):
        track = idx // (self.min_signal_len-self.seq_len)
        pos = idx % (self.min_signal_len-self.seq_len)

        outp = torch.tensor(self.data[track][pos:pos+self.seq_len])

        return outp[:-1], modules.a_law_encode(outp[-1:])[0]