# encoding: utf-8
import os
import glob
import random
import numpy as np
import librosa

class MyDataset():
    def __init__(self, folds, path):
        self.folds = folds
        self.path = path
        self.clean = 1 / 7.
        with open('../label_sorted.txt') as myfile:
            self.data_dir = myfile.read().splitlines()
        #self.filenames = glob.glob(os.path.join(self.path, '*', self.folds, '*.wav'))
        self.filenames = glob.glob(os.path.join(self.path, '*', self.folds, '*.npz'))
        self.list = {}
        for i, x in enumerate(self.filenames):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))

    def normalisation(self, inputs):
        inputs_std = np.std(inputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (inputs - np.mean(inputs))/inputs_std

    def __getitem__(self, idx):
        noise_prop = (1-self.clean)/6.
        temp = random.random()
        '''
        if self.folds == 'train':
            print(self.list[idx][0])
            print(self.list[idx][0][:36], self.list[idx][0][42:])
            if temp < noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/-5dB/'+self.list[idx][0][42:]
            elif temp < 2 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/0dB/'+self.list[idx][0][42:]
            elif temp < 3 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/5dB/'+self.list[idx][0][42:]
            elif temp < 4 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/10dB/'+self.list[idx][0][42:]
            elif temp < 5 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/15dB/'+self.list[idx][0][42:]
            elif temp < 6 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/20dB/'+self.list[idx][0][42:]
            else:
                self.list[idx][0] = self.list[idx][0]
        elif self.folds == 'val' or self.folds == 'test':
                self.list[idx][0] = self.list[idx][0]
        '''
        inputs = np.load(self.list[idx][0])['data']
        #print(self.list[idx])
        #inputs = librosa.load(self.list[idx][0], sr=16000)[0][-19456:]
        labels = self.list[idx][1]
        inputs = self.normalisation(inputs)
        #print('labels:', labels)
        return inputs, labels

    def __len__(self):
        return len(self.filenames)
