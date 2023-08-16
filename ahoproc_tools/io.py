"""
MIT License

Copyright (c) 2016 Santi Dsp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Here are the main options to interpolate Ahocoder features
(either can be lf0 or voided-frequency).
"""

from __future__ import print_function
from subprocess import run, PIPE
import numpy as np
import struct
import os

def read_aco_file(filename, out_shape=None):
    with open(filename, 'rb') as bs_f:
        fs_bs = bs_f.read()
    raw = struct.unpack('{}f'.format(int(len(fs_bs) / 4)), fs_bs)
    raw = np.array(raw, dtype=np.float32)
    if out_shape is not None:
        raw = raw.reshape(out_shape)
    return raw

def write_aco_file(filename, data):
    with open(filename, 'wb') as bs_f:
        # flatten all
        data = data.reshape((-1,))
        data_bs = struct.pack('%sf' % len(data), *data)
        bs_f.write(data_bs)

def aco2wav(basename, out_name=None, pitch_ext='.lf0'):
    # basename: acoustic file without cc, lf0 or fv extension
    cc_name = basename + '.cc'
    lf0_name = basename + pitch_ext
    fv_name = basename + '.fv'
    if out_name is None:
        wav_name = basename + '.wav'
    else:
        wav_name = out_name
    try:
        p = run(['ahodecoder16_64', lf0_name, cc_name, fv_name, wav_name],
                stdout=PIPE, 
                encoding='ascii')
    except FileNotFoundError:
        print('Please, make sure you have ahocoder16_64 binary in your $PATH')
        raise

def wav2aco(wav_name, out_name=None):
    # basename: acoustic file without cc, lf0 or fv extension
    bname = os.path.splitext(wav_name)[0]
    if out_name is None:
        aco_name = bname
    else:
        aco_name = out_name
    cc_name = aco_name + '.cc'
    lf0_name = aco_name + '.lf0'
    fv_name = aco_name + '.fv'
    try:
        p = run(['ahocoder16_64', wav_name, lf0_name, cc_name, fv_name],
                stdout=PIPE, 
                encoding='ascii')
        #print(p)
    except FileNotFoundError:
        print('Please, make sure you have ahocoder16_64 binary in your $PATH')
        raise
    return aco_name
