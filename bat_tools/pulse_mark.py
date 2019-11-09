# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:26:51 2019

@author: casim
"""
import matplotlib.pyplot as plt
from bat_tools.split_wav import split_wav
import os
import numpy as np

class PulseMark():

    def __init__(self, files, species, overwrite=False):
        self.d = {}
        self.files = files
        species_list = ['pipi', 'pypi', 'myo', 'plau', 'nyle', 'nyno', 'epse',
                        'rhhi', 'rhfe', 'baba', 'pina']

        assert species.lower() in species_list, f'Choose valid species from {species_list}'
        self.species = species
        self.curr_pos = 0
        self.curr_file = 0
        self.data = {files[0]:{}}

        # Check whether these files have associated text file
        complete=[]
        for file in self.files:
            txt = f'{os.path.splitext(file)[0]}-{self.species.upper()}.txt'
            if os.path.isfile(txt):

                complete.append(file)

        # Dealing with previously analysed data
        print(f'{len(self.files)} files selected for analysis.')
        print(f'{len(complete)} files have been previously analysed (and saved)')

        if not overwrite and complete != []:
            print(f'These files(s) will not be re-analysed - set overwrite=True to re-analyse.')
            self.files = [e for e in self.files if e not in complete]
        elif overwrite and complete != []:
            print(f'These files(s) will be re-analysed - set overwrite=False to skip these files.')


    def interact(self, n_fft=1024, ylim=(10, 150), save=True, cmap='viridis'):
        """
        n_fft should be related to the width of the bat call. In this 
        case the bat call duration is approx. 50ms (PIPI). So, to ensure 
        there are a decent resolution over this region, we will make 
        fft_window 50ms/25 = 2ms window. 2ms * 500000 = nfft = 1000. We 
        will use 1024 as this is more easily & repeatedly divisable by 2.
        """
        self.ylim=[i * 1000 for i in ylim]
        cmap=cmap

        def complex_click(event):
            """
            Simple clicking function, which records position of clicks (in x)
            and marks them on the interactive viewer.
            """
            # curr_file = self.curr_file
            if event.xdata is not None:
                self.data.setdefault(self.files[self.curr_file], {})# .append(event.xdata)
                self.data[self.files[self.curr_file]].setdefault(self.curr_pos, []).append(event.xdata)
                plt.axvline(event.xdata, color='k', lw=1)
                fig.canvas.draw()


        def complex_key(e, split, rate, t, freq, cmap, n_fft, files):
            """
            Deals with a range of key presses.

            Left/right scrolls forwards and backwards through the file
            Space will clear the saved data for that slice of data
            Esc will makr the file as bad (bad label?) and move to next file

            When moving right through the data, you first move through multiple
            slices making up file and then move onto each subsequent file.
            """

            if e.key == "right":
                self.curr_pos += 1
            elif e.key == "left":
                self.curr_pos -= 1
            elif e.key == ' ':
                self.data[self.files[self.curr_file]].pop(self.curr_pos, None)
            elif e.key == 'escape':
                self.curr_bool = False
                self.curr_pos = len(self.split)
                self.data.setdefault(self.files[self.curr_file], {})
                self.data[self.files[self.curr_file]] = {0:[False,]}
            else:
                return

            if self.curr_pos == len(self.split) and self.curr_file == len(files) - 1:
                self.curr_pos -=1
                self.d = {k: sum(self.data[k].values(), []) for k in self.data.keys()}

                ax.text(0.5, 0.9,'End of Files/Data', ha='center', va='center',
                        transform=ax.transAxes, bbox=dict(boxstyle='round', fc='w'))
                fig.canvas.draw()

            else:
                if self.curr_pos == len(self.split):
                    self.curr_file += 1
                    # print(self.data)
                    self.curr_pos = 0
                    self.curr_bool = True
                    self.split, self.t, freq, rate = split_wav(files[self.curr_file], n_fft=n_fft)
                    self.n_split = len(self.split)

                ax.cla()
                ax.pcolorfast([self.t[self.curr_pos][0], self.t[self.curr_pos][-1]],
                              [freq[0], freq[-1]], self.split[self.curr_pos], cmap=cmap)
                ax.set_yticks(self.ytick_loc)
                ax.set_yticklabels(self.yticks / 1000)
                plt.title(files[self.curr_file])
                ax.set_ylabel('kHz')
                ax.set_ylim(*self.ylim)
                ax.set_xlabel('time (s)')
                ax.text(0.9, 0.9,'{}/{}'.format(self.curr_pos+1, len(self.split)),
                        ha='center', va='center', transform=ax.transAxes,
                        bbox=dict(boxstyle='round', fc='w'))
                ax.set_ylim(*np.interp(self.ylim, freq, self.fz))
                ax.grid()
                fig.canvas.draw()

        def handle_close(evt):
            """
            Data is stored when the interactive screen is closed. Only completed
            files are stored, partially complete files are discarded.
            """
            if self.curr_pos < self.n_split - 1:
                self.data.pop(self.files[self.curr_file], None)
            self.d = {k: sum(self.data[k].values(), []) for k in self.data.keys()}


            
        # Now all the event handlers are ready, lets fire up some data
        print(f'Example file: {self.files[0]}')
        self.split, self.t, freq, rate = split_wav(self.files[0], n_fft=n_fft, hop_length=None)
        self.n_split = len(self.split)
        self.curr_bool = True
        fig, ax = plt.subplots()
        plt.title(self.files[0])

        ke = lambda x, args=[self.split, rate, self.t, freq, cmap, n_fft, self.files]: complex_key(x, *args)
        fig.canvas.mpl_connect('button_press_event', complex_click)
        fig.canvas.mpl_connect('key_press_event', ke)
        fig.canvas.mpl_connect('close_event', handle_close)

        ax.pcolorfast([self.t[0][0], self.t[0][-1]], [freq[0], freq[-1]], self.split[0], cmap=cmap) 
        ax.text(0.9, 0.9,'{}/{}'.format(self.curr_pos+1, len(self.split)), ha='center',
                va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', fc='w'),horizontalalignment='center',
                verticalalignment='center',)

        self.yticks=np.linspace(10000, 150000, 15)
        self.fz = np.linspace(freq[0], freq[-1], freq.size)
        self.ytick_loc = np.interp(self.yticks, freq, self.fz)

        ax.set_yticks(self.ytick_loc)
        ax.set_yticklabels(self.yticks / 1000)
        ax.set_ylabel('kHz')
        ax.set_xlabel('time (s)')
        ax.set_ylim(*np.interp(self.ylim, freq, self.fz))
        ax.grid()

        plt.show()

    def save(self):
        for file in self.d.keys():
            txt = f'{os.path.splitext(file)[0]}-{self.species.upper()}.txt'
            print(txt)
            with open(txt, 'w') as f:
                f.writelines("%s\n" % str(place) for place in self.d[file])

