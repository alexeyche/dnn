#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:15:34 2016

@author: alexeyche
"""

import numpy as np
import librosa as lr
import argparse
from lib.util import run_proc
from lib.util import setup_logging
from lib import run_iaf_network, write_time_series
from librosa.core.time_frequency import mel_frequencies

import logging
import os

setup_logging(logging.getLogger())


def main(input_file, ms_frame=10, n_mels=256, top_db=60, plot=False, fmin=100.0):
    input_file = os.path.realpath(os.path.expanduser(input_file))
    basef = input_file.rsplit(".",1)[0]
    logging.info("Processing {}".format(input_file))
    if input_file.endswith(".mp3"):
        logging.info("Got mp3 file, trying to convert")
        f = basef + ".wav"
        if not os.path.exists(f):
            run_proc(["mpg123", "-w", f, input_file])   
        input_file = f
    elif input_file.endswith(".wav"):
        pass
    else:
        raise Exception("Need wav or mp3 file to process")
        
    y, sr = lr.load(input_file)
    hop = int(np.round(ms_frame*sr/1000.0))
    stft = lr.stft(y, hop_length=hop)
    D = lr.logamplitude(np.abs(stft)**2, ref_power=np.max, top_db=top_db)
    s = lr.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, fmin=fmin, fmax=sr/2.0)
    mf = mel_frequencies(n_mels, fmin, sr/2.0)
    
    logging.info("Normalizing between 0 and 1")
    for ri in xrange(s.shape[0]):    
        denom = np.max(s[ri,:]) - np.min(s[ri,:])
        if denom > 0:
            s[ri,:] = (s[ri,:] - np.min(s[ri,:]))/denom
        else:
            s[ri,:] = 0
            
    if plot:
        from matplotlib import pyplot as plt
        
        plt.figure(1)
        plt.subplot(2,1,1)
        lr.display.specshow(
            lr.logamplitude(
                    lr.stft(np.abs(y))**2, 
                ref_power=np.max, top_db=top_db), 
            sr, y_axis="mel")
        plt.subplot(2,1,2)
        lr.display.specshow(s, sr, hop_length=hop)
        plt.show()

    write_time_series(s, basef+"_raw.pb")
    run_iaf_network(s, basef+".pb", dt=1.0, tau_mem=10, tau_ref=2, threshold=0.4)
    
    return s
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracting mel spectrogram from file')
    parser.add_argument('file', nargs=1, help="Path to music file")
    parser.add_argument('--ms-frame', help='Ms in one frame', type=int, default=10)
    parser.add_argument('--n-mels', help='Number of mel components', type=int, default=256)
    parser.add_argument('--top-db', help='Top dB', type=int, default=60)
    parser.add_argument('--plot',action='store_true', help='Debug spectrogram plot')
    args = parser.parse_args()
    
    main(args.file[0], args.ms_frame, args.n_mels, args.top_db, args.plot)
    
    
    
