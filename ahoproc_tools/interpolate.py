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

import argparse
import struct
import os

import numpy as np


def linear_interpolation(tbounds, fbounds):
    """Linear interpolation between the specified bounds"""
    interp = []
    for t in range(tbounds[0], tbounds[1]):
        interp.append(fbounds[0] + (t - tbounds[0]) * ((fbounds[1] - fbounds[0]) /
                                                       (tbounds[1] - tbounds[0])))
    return interp


def interpolation(signal, unvoiced_symbol):
    tbound = [None, None]
    fbound = [None, None]
    signal_t_1 = signal[0]
    isignal = np.copy(signal)
    uv = np.ones(signal.shape, dtype=np.int8)
    for t in range(1, signal.shape[0]):
        if (signal[t] > unvoiced_symbol) and (signal_t_1 <= unvoiced_symbol) and (tbound == [None, None]):
            # First part of signal is unvoiced, set to constant first voiced
            isignal[:t] = signal[t]
            uv[:t] = 0
        elif (signal[t] <= unvoiced_symbol) and (signal_t_1 > unvoiced_symbol):
            tbound[0] = t - 1
            fbound[0] = signal_t_1
        elif (signal[t] > unvoiced_symbol) and (signal_t_1 <= unvoiced_symbol):
            tbound[1] = t
            fbound[1] = signal[t]
            isignal[tbound[0]:tbound[1]] = linear_interpolation(tbound, fbound)
            uv[tbound[0]:tbound[1]] = 0
            # reset values
            tbound = [None, None]
            fbound = [None, None]
        signal_t_1 = signal[t]
    # now end of signal if necessary
    if tbound[0] is not None:
        isignal[tbound[0]:] = fbound[0]
        uv[tbound[0]:] = 0
    # if all are unvoiced symbols, uv is zeros
    if np.all(isignal <= unvoiced_symbol):
        uv = np.zeros(signal.shape, dtype=np.int8)
    return isignal, uv


def process_file(filename, unvoiced_symbol, gen_uv, bin_mode=False):
    dire, fullname = os.path.split(filename.rstrip())
    basename, ext = os.path.splitext(fullname)
    if bin_mode:
        # read raw floats from bitstream
        with open(filename, 'rb') as bs_f:
            fs_bs = bs_f.read()
        raw = struct.unpack('{}f'.format(int(len(fs_bs) / 4)), fs_bs)
        raw = np.array(raw, dtype=np.float32)
    else:
        # load floats in txt format
        raw = np.loadtxt(filename)
    interp, uv = interpolation(raw, unvoiced_symbol)
    out_interp_file = os.path.join(dire, basename + '.i' + ext)
    print('Writing interpolation to {}'.format(out_interp_file))
    if bin_mode:
        # write raw floats into bitstream
        interp_bs = struct.pack('%sf' % len(interp), *interp)
        with open(out_interp_file, 'wb') as interp_f:
            interp_f.write(interp_bs)
    else:
        np.savetxt(out_interp_file, interp)
    if gen_uv:
        out_uv_file = os.path.join(dire, basename + '.uv')
        print('Writing u/v mask to {}'.format(out_uv_file))
        if bin_mode:
            # write raw floats into bitstream
            uv_bs = struct.pack('%sf' % len(uv), *uv)
            with open(out_uv_file, 'wb') as uv_f:
                uv_f.write(uv_bs)
        else:
            np.savetxt(out_uv_file, uv)


def process_guia(guia_file, unvoiced_symbol, gen_uv, bin_mode=False):
    # Interpolate files values
    with open(guia_file) as fh:
        for i, filename in enumerate(fh):
            process_file(filename.rstrip(), unvoiced_symbol, gen_uv, bin_mode)


def main(opts):
    if opts.f0_file:
        process_file(opts.f0_file, -10000000000, opts.gen_uv, opts.bin_mode)
    if opts.f0_guia:
        process_guia(opts.f0_guia, -10000000000, opts.gen_uv, opts.bin_mode)
    if opts.vf_file:
        process_file(opts.vf_file, 1e3, opts.gen_uv, opts.bin_mode)
    if opts.vf_guia:
        process_guia(opts.vf_guia, 1e3, opts.gen_uv, otps.bin_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Here are the main options to interpolate'
                                     ' Ahocoder features')
    parser.add_argument('--f0_guia', type=str,
                        default=None, help='Guia file containing pointers to '
                                           'the different lf0 files to '
                                           'interpolate.')
    parser.add_argument('--f0_file', type=str,
                        default=None, help='Filename of a single F0 file')
    parser.add_argument('--vf_guia', type=str,
                        default=None, help='Guia file containing pointers to '
                                           'the different vf files to '
                                           'interpolate.')
    parser.add_argument('--vf_file', type=str,
                        default=None, help='Filename of a single VF file')
    parser.add_argument('--no-uv', dest='gen_uv',
                        action='store_false', help='U/V masks are NOT '
                                                   'generated.')
    parser.add_argument('--bin_mode', action='store_true', default=False,
                        help='Work with binary acoustic files, not txt.')
    parser.set_defaults(gen_uv=True)
    options = parser.parse_args()
    main(options)
