# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:28:08 2019

@author: casim
"""
import numpy as np
import librosa
import os
import itertools

species_list = ['PIPI', 'NYNO', 'MYO', 'RHHI', 'EPSE', 'PYPI', 'BABA', 'PLAU',
                'NYLE', 'RHFE', 'PINA'] ## add the rest


def load_split_wav(f, n_fft=2048, n_mels=256, hop_length=None, fmin=10000, fmax=150000):
    """
    Loads wav file, converts to mel spectogram that has been split into short 
    (0.5s) sections for the purpose of markinng the pulse locations.
    
    Parameters
    ----------
    f: str
        The path to the wav file
    n_fft: int
        Length of FFT window
    n_mels: int
        Number of bins for the frequency data (using a mel filter bank 
        construction)
    hop_length: int/None
        Number of samples between successive frames.
    fmin: int
        Minimum frequency for mel bins
    fmax: int
        Maximum frequency for mel bins
    

    Returns
    -------
    rate: int
        Sampling rate
    split: array
        Array containing all n logmel spectograms for all pulses in specified 
        file, of size (n , window, n_mels)
    t: array
        Time vector across each windows for each split
    freq: array
        Frequencies associated with the n_mel frequency bands

    """
    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(n_fft // 4)

    (sig, rate) = librosa.load(f, sr=None)
    melspec = librosa.feature.melspectrogram(sig, n_fft=2048, n_mels=n_mels, hop_length=hop_length, 
                                             fmin=fmin, fmax=fmax, sr=rate)
    logmel = librosa.core.power_to_db(melspec)
    t_total = sig.size/rate
    
    
    split_num = (t_total / 0.5)
    split = np.array_split(logmel, split_num, axis=1)
    t = np.linspace(0, (sig.size/rate), logmel.shape[1])
    t = np.array_split(t, split_num)
    
    f = librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    return rate, split, t, f
    
def closest_argmin(A, B):
    """
    For each element in A finds the index of the closest value in array B.
    
    Parameters
    ----------
    A: array
        Array with elements to search for
    B: array
        Array to search within
  

    Returns
    -------
    rate: array
        List of closest indices
    """
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    ind_list = sidx_B[sorted_idx-mask]
    return ind_list


def split_save(txt, window=256, n_fft=2048, n_mels=256, hop_length=None, fmin=10000, fmax=150000):
    """
    Split wav at time points defined in associated txt file. Split data is
    the melspectogram across the window of interest.
    
    Parameters
    ----------
    txt: str
        The path to the folder structure to walk through
    window: int
        The number of points to take around the defined pulse (i.e time wrt 
        sampling rate)
    n_fft: int
        Length of FFT window
    n_mels: int
        Number of bins for the frequency data (using a mel filter bank 
        construction)
    hop_length: int/None
        Number of samples between successive frames.
    fmin: int
        Minimum frequency for mel bins
    fmax: int
        Maximum frequency for mel bins
    

    Returns
    -------
    split: array
        Array containing all n logmel spectograms for all pulses in specified 
        file, of size (n , window, n_mels)
    t_split: array
        Time vector across each windows for each pulse, with size (n, window)
    freq: array
        Frequencies associated with the n_mel frequency bands

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

def txt_list(path):
    """
    Walk through folder finding all analyzed files (i.e. wav files with associated
    text files)
    
    Parameters
    ----------
    path: str
        The path to the folder structure to walk through

    Returns
    -------
    files: list
        List of the absolute path to all the found text files (skips empty and
        those labelled false i.e. incorrect labelling)
    species_labels: list
        List of species labels associated with those files
    counts: list
        Number of pulses within each file
    min_space: list
        The minimum spacing between pulses. For error checking (i.e. spacing
        should be > 0)
    """
    files = []
    counts = []
    min_space = []
    species_labels = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file and os.path.splitext(file)[0].split('-')[-1].upper() in species_list:
                fpath = os.path.join(r, file)
                try:
                    times = np.sort(np.loadtxt(fpath))
                    
                    min_space.append(np.min(times[1:] - times[:-1]))
                    counts.append(times.size)
                    files.append(fpath)
                    species_labels.append(os.path.splitext(file)[0].split('-')[-1].upper())
                except ValueError: # if False stored in file
                    pass
                
    return files, species_labels, counts, min_space


def split_save_path(path):
    """
    Walks through directory, finds all text files associated with pulse marked 
    wav files, locates pulses and extracts logmel spectograms (256 x 256) for
    each pulse. This window will cover > 1 pulse for most bat species.
    
    Parameters
    ----------
    path: str
        The path to the folder structure to walk through

    Returns
    -------
    d: array
        Array containing all n logmel spectograms for all pulses across all 
        files found in the folder structure of size (n , 256, 256)
    sp_all: array
        List of species associated with the dataset d, of size (n,)
    """
    files, species_labels, counts, min_space = txt_list(path)
    data = []
    sp_all = []
    for f, sp in zip(files, species_labels):
        X = split_save(f)
        if X != None:
            sp_all.append(X[0].shape[0] * [sp,])
            data.append(X[0])
    sp_all = np.array(list(itertools.chain(*sp_all)))
    d = np.vstack(data)
    return d, sp_all