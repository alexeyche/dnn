# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 18:53:49 2016

@author: alexeyche
"""

import numpy as np
import librosa as lr
import logging
from os.path import expanduser as pp
from matplotlib import pyplot as plt

def gauss(x, mean, sd):
    return np.exp( - (x - mean)*(x - mean)/sd)

def build_filter(mixm, sr, gauss_size, n_fft):
    n_mels, n = mixm.shape

    fftf = lr.fft_frequencies(sr=sr, n_fft=n_fft)
    melf = lr.mel_frequencies(fmin=100.0, n_mels = n_mels)
    
    n_fft_work = fftf.shape[0]
    
    filt = np.zeros((n, n_fft_work))
    for mi in xrange(n_mels):
        freq = melf[mi]
        df = np.abs(fftf - freq)
        fft_id = np.where(df == np.min(df))[0]
        if gauss_size == 1:
            filt[:, fft_id] += np.asarray([mixm[mi, :]]).T
        else:
            for filt_id in xrange(fft_id - gauss_size/2, fft_id + gauss_size/2):
                if filt_id >= n_fft_work:
                    continue
                filt[:, filt_id] += mixm[mi, :] * gauss(filt_id, fft_id, sd=1)
    return filt

def process(stft, filt_vec):
    stft_res = np.ndarray(stft.shape, dtype=stft.dtype)    
    for frame_id in xrange(stft.shape[1]):
        stft_res[:, frame_id] = filt_vec * stft[:, frame_id]
        
    y_r = lr.istft(stft_res, hop_length=hop)
    return stft_res, y_r
    
input_file = pp("~/Music/ml/test_licks.wav")

y, sr = lr.load(input_file)

n_fft = 2048 # default
gauss_size = 1


ms_frame = 10
hop = int(np.round(ms_frame*sr/1000.0))
stft = np.abs(lr.stft(y, hop_length=hop, n_fft = n_fft))

mixm = np.loadtxt(pp("~/dnn/runs/ica_filter.csv"), delimiter=",")

_, n = mixm.shape

filt = build_filter(mixm, sr, gauss_size, n_fft)

for ni in xrange(n):
    print ni
    stft_res, y_r = process(stft, filt[ni,])
    
    plt.figure(1)
    plt.subplot(2,1,1)
    lr.display.specshow(
        lr.logamplitude(
                stft**2, 
            ref_power=np.max, top_db=100), 
        sr, y_axis="mel")
    plt.subplot(2,1,2)
    lr.display.specshow(
        lr.logamplitude(
                stft_res**2, 
            ref_power=np.max, top_db=100), 
        sr, y_axis="mel")
    plt.savefig(pp("~/Music/ml/{}_feat.png".format(ni)))
    
    lr.output.write_wav(pp("~/Music/ml/{}_feat.wav".format(ni)), y_r, sr)
    
y = lr.istft(stft, hop_length=hop)
lr.output.write_wav(pp("~/Music/ml/real_feat.wav"), y, sr)