import numpy as np
import os
import itertools

from bat_tools.split_wav import logmel_wav

species_list = ['PIPI', 'PIPY', 'PINA', 
                'NYNO', 'NYLE', 'EPSE', 
                'MYO', 'BABA', 'PLAU',
                'RHHI', 'RHFE']


def slider(f, step=64, window_width=256, n_fft=2048, n_mels=256, 
           hop_length=None, fmin=10000, fmax=150000):
    """
    Loads wav file, converts to mel spectogram that has been split into short, 
    defined windows with a (potentially overlapping) step.
    
    Parameters
    ----------
    f: str
        The path to the wav file
    step: int
        Sliding window step
    window_width: 256
        Window width for slider
    n_fft: int
        Length of FFT window
    n_mels: int
        Number of bins for the frequency data (using a mel filter bank 
        construction)
    hop_length: int/None
        Number of samples between successive frames.
    fmin: int
        Minimum frequency for mel bins (Hz)
    fmax: int
        Maximum frequency for mel bins (Hz)
    

    Returns
    -------
    slide: array
        Array containing all n logmel spectograms for sliding window
        appplied to load wav (n , window, n_mels)
    t: array
        Time vector across each window for each section
    freq: array
        Frequencies associated with the n_mel frequency bands
    rate: int
        Sampling rate
    """

    logmel, t, freq, rate = logmel_wav(f, n_fft, n_mels, hop_length, fmin, fmax)

    indexes = np.arange(0, logmel.shape[1] - window_width, step)
    window = np.arange(window_width)
    slide = np.repeat(window[None, :], indexes.size, axis=0)
    slide += indexes[:, None]

    return np.moveaxis(logmel[:, slide], 1, 0), t[slide], freq, rate