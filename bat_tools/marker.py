# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:38:23 2019

@author: casim
"""
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()


parser.add_argument("-f", dest="fpath",
                    help="Path to csv with file list", type=str)

parser.add_argument("-s", dest="species",
                    help="Species", type=str)

parser.add_argument("-n", dest="n_fft",
                    help="Number of fft (default 1024)", type=int,
                    default=1024)

parser.add_argument("--ymin", dest="ymin",
                    help="ymin (default 10)", type=int,
                    default=10)

parser.add_argument("--ymax", dest="ymax",
                    help="ymax (default 150)", type=int,
                    default=150)

parser.add_argument("--overwrite", dest="overwrite", action='store_true',
                    help="Analyse previously marked datafiles", default=False)

parser.add_argument('--no-save', dest='save', default=True, action='store_false',
                    help="Do not save text file with list of pulse positions")


args = parser.parse_args()

from bat_tools.pulse_mark import PulseMark
files = np.genfromtxt(args.fpath, dtype=str,  delimiter=',')
A = PulseMark(files, args.species, args.overwrite)
A.interact(args.n_fft, [args.ymin, args.ymax])
if args.save:
    A.save()