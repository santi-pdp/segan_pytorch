from subprocess import run, PIPE
from scipy.io import wavfile
import numpy as np
import tempfile
import os
import re

def denormalize_wave_minmax(x):
    return (65535. * x / 2) - 1 + 32767.

def PESQ(ref_wav, deg_wav):
    # reference wav
    # degraded wav

    tfl = tempfile.NamedTemporaryFile()
    ref_tfl = tfl.name + '_ref.wav'
    deg_tfl = tfl.name + '_deg.wav'

    if ref_wav.max() <= 1:
        ref_wav = np.array(denormalize_wave_minmax(ref_wav), dtype=np.int16)
    if deg_wav.max() <= 1:
        deg_wav = np.array(denormalize_wave_minmax(deg_wav), dtype=np.int16)
	
    wavfile.write(ref_tfl, 16000, ref_wav)
    wavfile.write(deg_tfl, 16000, deg_wav)
    
    curr_dir = os.getcwd()
    # Write both to tmp files and then eval with pesqmain
    try:
        p = run(['pesqmain'.format(curr_dir), 
                 ref_tfl, deg_tfl, '+16000'],
                stdout=PIPE, 
                encoding='ascii')
        res_line = p.stdout.split('\n')[-2]
        results = re.split('\s+', res_line)
        return results[-1]
    except FileNotFoundError:
        print('pesqmain not found! Please add it your PATH')


def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    ref_len = ref_wav.shape[0]
    deg_wav = deg_wav[:len_]

    # scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / np.sum(dif ** 2))

    # global variables
    winlen = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate = winlen // 4
    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR

    num_frames = ref_len / skiprate - (winlen/skiprate)
    start = 0
    t = np.linspace(1, winlen, winlen) / (winlen + 1)
    window = 0.5*(1 - np.cos(2 * np.pi * t))
    segmental_snr = []
    for idx in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        ref_frame = ref_wav[idx:idx + winlen]
        deg_frame = deg_wav[idx:idx + winlen]
        ref_frame = ref_frame * window
        deg_frame = deg_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(ref_frame ** 2)
        noise_energy = np.sum((ref_frame - deg_frame) ** 2)
        ssnr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
        ssnr = max(ssnr, MIN_SNR)
        ssnr = min(ssnr, MAX_SNR)
        segmental_snr.append(ssnr)
    snr_mean = overall_snr
    segsnr_mean = np.mean(segmental_snr)
    return snr_mean, segsnr_mean
