# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:28:08 2019

@author: casim
"""
import numpy as np
import librosa

def load_split_wav(f, n_fft=2048, hop_length=None, dt=0.45):
    
    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(n_fft // 4)

    (sig, rate) = librosa.load(f, sr=None)
    S = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length,))
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    t_total = sig.size/rate
    
    
    split_num = (t_total / 0.5)
    split = np.array_split(log_S, split_num, axis=1)
    t = np.linspace(0, (sig.size/rate), log_S.shape[1])
    t = np.array_split(t, split_num)
    f = np.linspace(0, rate/2000, log_S.shape[0])
    return rate, split, t, f
    