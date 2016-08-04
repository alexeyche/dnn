# -*- coding: utf-8 -*-
"""
Created on Mon May  9 05:47:06 2016

@author: alexeyche
"""

import numpy as np
import librosa
import os
from matplotlib import pyplot as plt

os.chdir(os.path.expanduser("~/Music/ml/"))

song = "Rose Room"

files = [ f for f in os.listdir(".") if f.endswith("wav") and song in f ]

y, sr = librosa.load(files[0])

hop = 64
n_mels = 256
top_db = 60

stft = librosa.stft(y, hop_length=hop)
D = librosa.logamplitude(np.abs(stft)**2, ref_power=np.max, top_db=top_db)
s = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels)

plt.figure(1)

plt.subplot(2,1,1)
librosa.display.specshow(
    librosa.logamplitude(
        librosa.stft(np.abs(y))**2, 
    ref_power=np.max, top_db=top_db), 
sr, y_axis="mel")

plt.subplot(2,1,2)
librosa.display.specshow(s, sr, hop_length=hop)

