'''
Python module for 5G NR sync signals and decoding.
Copyright 2021 Mark Disterhof.
'''

import sys
from os.path import dirname
sys.path.append(dirname(__file__))

import ssb
import signals
import decode

__version__ = "0.9.0"
__author__ = 'Mark Disterhof'