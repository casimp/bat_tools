# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:50:02 2019

@author: casim
"""
import os
import re
import pandas as pd
import numpy as np


def scrape_meta_wav(files, species=True):

    sp, T, time, loc, make, model, fs = [], [], [], [], [], [], []
    for idx, f in enumerate(files):
        if idx % 500 == 0:
            print(idx)

        with open(f, "rb") as binary_file:
            # print(i, files[i])
            # Read the whole file at once
            binary_file.seek(-500, 2)
            data = binary_file.read()
            try:
                if species:
                    sp.append(re.search(b':(.*)\n', data[data.rfind(b'Species Manual ID'):]).group(1).decode('ascii'))
                else:
                    sp.append(None)
                T.append(float(re.search(b':(.*)\n', data[data.rfind(b'Temperature Int'):]).group(1).decode('ascii')))
                time.append(re.search(b':(.*)\n', data[data.rfind(b'Timestamp'):]).group(1).decode('ascii'))
                loc.append(re.search(b':(.*)\n', data[data.rfind(b'Loc Position'):]).group(1).decode('ascii'))
                make.append(re.search(b':(.*)\n', data[data.rfind(b'Make'):]).group(1).decode('ascii'))
                model.append(re.search(b':(.*)\n', data[data.rfind(b'Model'):]).group(1).decode('ascii'))
                fs.append(f)
            except AttributeError:
                pass

    df = pd.DataFrame(data = np.column_stack([sp, T, time, loc, make, model]),
                  columns=['Species', 'Temperature', 'time', 'Loc', 'Make', 'Model'], index=fs)
    return df

def scrape_meta_wav_folder(folder, species=True):

    files = []
    # r=root, d=directories, f = files
    for r, _, f in os.walk(folder):
        for file in f:
            if '.wav' in file:
                files.append(os.path.join(r, file))

    print(f'{len(files)} files')

    return scrape_meta_wav(files, species)

def scrape_meta_wav_files(files, species=True):

    print(f'{len(files)} files')

    return scrape_meta_wav(files, species)

def find(name, path):
    for root, _, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def scrape_labels_zc(zc_base, wav_base):
    """
    Used to scrape ZC data created by S.
    """
    months = [i for i in os.listdir(zc_base) if os.path.isdir(os.path.join(zc_base, i))]

    dfs = []
    for month in months:
        m_path = os.path.join(zc_base, month)
        m_path_wav = os.path.join(wav_base, month)
        outputs = [i for i in os.listdir(m_path) if 'output' in i]

        for output in outputs:
            o_path = os.path.join(m_path, output)
            df = pd.read_csv(o_path, delimiter='\t', usecols=['Filename', 'Label'])
            df['wav'] = df['Filename'].str.replace('_00000_000.00#',  '.wav')
            df['wav'] = [find(f,m_path_wav) for f in df['wav']]
            df['zc'] = [find(f,m_path) for f in df['Filename']]

            dfs.append(df)
            df_all = pd.concat(dfs)
    return df_all

