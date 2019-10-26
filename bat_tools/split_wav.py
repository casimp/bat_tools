# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:28:08 2019

@author: casim
"""
import numpy as np
import librosa
import os

def load_split_wav(f, n_fft=2048, n_mels=256, hop_length=None, fmin=10000, fmax=150000):
    
    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(n_fft // 4)

    (sig, rate) = librosa.load(f, sr=None)
    melspec = librosa.feature.melspectrogram(sig, n_fft=2048, n_mels=n_mels, hop_length=hop_length, 
                                             fmin=fmin, fmax=fmax, sr=rate)
    logmel = librosa.core.power_to_db(melspec)
    
    
    
    # (sig, rate) = librosa.load(f, sr=None)
    # S = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_length,))
    # log_S = librosa.amplitude_to_db(S, ref=np.max)
    t_total = sig.size/rate
    
    
    split_num = (t_total / 0.5)
    split = np.array_split(logmel, split_num, axis=1)
    t = np.linspace(0, (sig.size/rate), logmel.shape[1])
    t = np.array_split(t, split_num)
    # f = np.linspace(0, rate/2000, logmel.shape[0])
    
    f = librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    return rate, split, t, f
    
def closest_argmin(A, B):
    """
    Find A within B
    """
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]


def split_save(txt, window=256, n_fft=2048, n_mels=256, hop_length=None, fmin=10000, fmax=150000):
    """
    Split at defined locations
    """
    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(n_fft // 4)
        
    # species = os.path.splitext(txt)[0].rsplit('-', 1)[1]
    wav = f"{os.path.splitext(txt)[0].rsplit('-', 1)[0]}.wav"

    (sig, rate) = librosa.load(wav, sr=None)
    melspec = librosa.feature.melspectrogram(sig, n_fft=2048, n_mels=n_mels, hop_length=hop_length, 
                                             fmin=fmin, fmax=fmax, sr=rate)
    logmel = librosa.core.power_to_db(melspec)
    # print(logmel.shape)
    t = np.linspace(0, (sig.size/rate), logmel.shape[1])
    freq = librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    
    # txt = f'{os.path.splitext(f)[0]}-{species.upper()}.txt'

    try:
        t_pulse = np.loadtxt(txt)
        assert t_pulse != np.array([]), 'Zero array'
        indexes = closest_argmin(t_pulse, t)
        window = np.arange(-window//2, window//2)
        split_range = np.repeat(window[None, :], indexes.size, axis=0) + indexes[:, None]
        split_range = split_range[~np.any(split_range < 0, axis=1)]
        split_range = split_range[~np.any(split_range > logmel.shape[1], axis=1)]

        split = logmel[:, split_range]
        t_split = t[split_range]

        return np.moveaxis(split, 1, 0), t_split, freq
    
    except (AssertionError, ValueError):
        pass
#    