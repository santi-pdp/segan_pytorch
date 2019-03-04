from subprocess import run, PIPE
from scipy.linalg import toeplitz
from scipy.io import wavfile
import numba as nb
from numba import jit, int32, float32
import soundfile as sf
from scipy.signal import lfilter
from scipy.interpolate import interp1d
import torch
import torch.nn.functional as F
import glob
import librosa
import numpy as np
import tempfile
import os
import re


def uttname2spkid(uttname):
    spkid = uttname.split('_')[0]
    return spkid

def denormalize_wave_minmax(x):
    return (65535. * x / 2) - 1 + 32767.

def make_divN(tensor, N, method='zeros'):
    # methods: zeros or reflect
    # make tensor time dim divisible by N (for good decimation)
    pad_num = (tensor.size(1) + N) - (tensor.size(1) % N) - tensor.size(1)
    if method == 'zeros':
        pad = torch.zeros(tensor.size(0), pad_num, tensor.size(-1))
        return torch.cat((tensor, pad), dim=1)
    elif method == 'reflect':
        tensor = tensor.transpose(1, 2)
        # using functional PyTorch pad
        return F.pad(tensor, (0, pad_num), 'reflect').transpose(1, 2)
    else:
        raise TypeError('Unrecognized make_divN pad method: ', method)

def composite_helper(args):
    return eval_composite(*args)

class ComposeAdditive(object):

    def __init__(self, additive):
        self.additive = additive

    def __call__(self, x):
        return x, self.additive(x)

class Additive(object):

    def __init__(self, noises_dir, snr_levels=[0, 5, 10], do_IRS=False):
        self.noises_dir = noises_dir
        self.snr_levels = snr_levels
        self.do_IRS = do_IRS
        # read noises in dir
        noises = glob.glob(os.path.join(noises_dir, '*.wav'))
        if len(noises) == 0:
            raise ValueError('[!] No noises found in {}'.format(noises_dir))
        else:
            print('[*] Found {} noise files'.format(len(noises)))
            self.noises = []
            for n_i, npath in enumerate(noises, start=1):
                #nwav = wavfile.read(npath)[1]
                nwav = librosa.load(npath, sr=None)[0]
                self.noises.append({'file':npath, 
                                    'data':nwav.astype(np.float32)})
                log_noise_load = 'Loaded noise {:3d}/{:3d}: ' \
                                 '{}'.format(n_i, len(noises),
                                             npath)
                print(log_noise_load)
        self.eps = 1e-22

    def __call__(self, wav, srate=16000, nbits=16):
        """ Add noise to clean wav """
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
        noise_idx = np.random.choice(list(range(len(self.noises))), 1)
        sel_noise = self.noises[np.asscalar(noise_idx)]
        noise = sel_noise['data']
        snr = np.random.choice(self.snr_levels, 1)
        # print('Applying SNR: {} dB'.format(snr[0]))
        if wav.ndim > 1:
            wav = wav.reshape((-1,))
        noisy, noise_bound = self.addnoise_asl(wav, noise, srate, 
                                               nbits, snr, 
                                               do_IRS=self.do_IRS)
        # normalize to avoid clipping
        if np.max(noisy) >= 1 or np.min(noisy) < -1:
            small = 0.1
            while np.max(noisy) >= 1 or np.min(noisy) < -1:
                noisy = noisy / (1. + small)
                small = small + 0.1
        return torch.FloatTensor(noisy.astype(np.float32))


    def addnoise_asl(self, clean, noise, srate, nbits, snr, do_IRS=False):
        if do_IRS:
            # Apply IRS filter simulating telephone 
            # handset BW [300, 3200] Hz
            clean = self.apply_IRS(clean, srate, nbits)
        Px, asl, c0 = self.asl_P56(clean, srate, nbits)
        # Px is active speech level ms energy
        # asl is active factor
        # c0 is active speech level threshold
        x = clean
        x_len = x.shape[0]

        noise_len = noise.shape[0]
        if noise_len <= x_len:
            print('Noise length: ', noise_len)
            print('Speech length: ', x_len)
            raise ValueError('Noise length has to be greater than speech '
                             'length!')
        rand_start_limit = int(noise_len - x_len + 1)
        rand_start = int(np.round((rand_start_limit - 1) * np.random.rand(1) \
                                  + 1))
        noise_segment = noise[rand_start:rand_start + x_len]
        noise_bounds = (rand_start, rand_start + x_len)

        if do_IRS:
            noise_segment = self.apply_IRS(noise_segment, srate, nbits)

        Pn = np.dot(noise_segment.T, noise_segment) / x_len

        # we need to scale the noise segment samples to obtain the 
        # desired SNR = 10 * log10( Px / ((sf ** 2) * Pn))
        sf = np.sqrt(Px / Pn / (10 ** (snr / 10)))
        noise_segment = noise_segment * sf
    
        noisy = x + noise_segment

        return noisy, noise_bounds

    def apply_IRS(self, data, srate, nbits):
        """ Apply telephone handset BW [300, 3200] Hz """
        raise NotImplementedError('Under construction!')
        from pyfftw.interfaces import scipy_fftpack as fftw
        n = data.shape[0]
        # find next pow of 2 which is greater or eq to n
        pow_of_2 = 2 ** (np.ceil(np.log2(n)))

        align_filter_dB = np.array([[0, -200], [50, -40], [100, -20],
                           [125, -12], [160, -6], [200, 0],
                           [250, 4], [300, 6], [350, 8], [400, 10],
                           [500, 11], [600, 12], [700, 12], [800, 12],
                           [1000, 12], [1300, 12], [1600, 12], [2000, 12],
                           [2500, 12], [3000, 12], [3250, 12], [3500, 4],
                           [4000, -200], [5000, -200], [6300, -200], 
                           [8000, -200]]) 
        print('align filter dB shape: ', align_filter_dB.shape)
        num_of_points, trivial = align_filter_dB.shape
        overallGainFilter = interp1d(align_filter_dB[:, 0], align_filter[:, 1],
                                     1000)

        x = np.zeros((pow_of_2))
        x[:data.shape[0]] = data

        x_fft = fftw.fft(x, pow_of_2)

        freq_resolution = srate / pow_of_2

        factorDb = interp1d(align_filter_dB[:, 0],
                            align_filter_dB[:, 1],
                                           list(range(0, (pow_of_2 / 2) + 1) *\
                                                freq_resolution)) - \
                                           overallGainFilter
        factor = 10 ** (factorDb / 20)

        factor = [factor, np.fliplr(factor[1:(pow_of_2 / 2 + 1)])]
        x_fft = x_fft * factor

        y = fftw.ifft(x_fft, pow_of_2)

        data_filtered = y[:n]
        return data_filtered


    def asl_P56(self, x, srate, nbits):
        """ ITU P.56 method B. """
        T = 0.03 # time constant of smoothing in seconds
        H = 0.2 # hangover time in seconds
        M = 15.9

        # margin in dB of the diff b/w threshold and active speech level
        thres_no = nbits - 1 # num of thresholds, for 16 bits it's 15

        I = np.ceil(srate * H) # hangover in samples
        g = np.exp( -1 / (srate * T)) # smoothing factor in envelop detection
        c = 2. ** (np.array(list(range(-15, (thres_no + 1) - 16))))
        # array of thresholds from one quantizing level up to half the max
        # code, at a step of 2. In case of 16bit: from 2^-15 to 0.5
        a = np.zeros(c.shape[0]) # activity counter for each level thres
        hang = np.ones(c.shape[0]) * I # hangover counter for each level thres

        assert x.ndim == 1, x.shape
        sq = np.dot(x, x) # long term level square energy of x
        x_len = x.shape[0]

        # use 2nd order IIR filter to detect envelope q
        x_abs = np.abs(x)
        p = lfilter(np.ones(1) - g, np.array([1, -g]), x_abs)
        q = lfilter(np.ones(1) - g, np.array([1, -g]), p)

        for k in range(x_len):
            for j in range(thres_no):
                if q[k] >= c[j]:
                    a[j] = a[j] + 1
                    hang[j] = 0
                elif hang[j] < I:
                    a[j] = a[j] + 1
                    hang[j] = hang[j] + 1
                else:
                    break
        asl = 0
        asl_ms = 0
        c0 = None
        if a[0] == 0:
            return asl_ms, asl, c0
        else:
            den = a[0] + self.eps
            AdB1 = 10 * np.log10(sq / a[0] + self.eps)
        
        CdB1 = 20 * np.log10(c[0] + self.eps)
        if AdB1 - CdB1 < M:
            return asl_ms, asl, c0
        AdB = np.zeros(c.shape[0])
        CdB = np.zeros(c.shape[0])
        Delta = np.zeros(c.shape[0])
        AdB[0] = AdB1
        CdB[0] = CdB1
        Delta[0] = AdB1 - CdB1

        for j in range(1, AdB.shape[0]):
            AdB[j] = 10 * np.log10(sq / (a[j] + self.eps) + self.eps)
            CdB[j] = 20 * np.log10(c[j] + self.eps)

        for j in range(1, Delta.shape[0]):
            if a[j] != 0:
                Delta[j] = AdB[j] - CdB[j]
                if Delta[j] <= M:
                    # interpolate to find the asl
                    asl_ms_log, cl0 = self.bin_interp(AdB[j],
                                                      AdB[j - 1],
                                                      CdB[j],
                                                      CdB[j - 1],
                                                      M, 0.5)
                    asl_ms = 10 ** (asl_ms_log / 10)
                    asl = (sq / x_len ) / asl_ms
                    c0 = 10 ** (cl0 / 20)
                    break
        return asl_ms, asl, c0

    def bin_interp(self, upcount, lwcount, upthr, lwthr, Margin, tol):
        if tol < 0:
            tol = -tol

        # check if extreme counts are not already the true active value
        iterno = 1
        if np.abs(upcount - upthr - Margin) < tol:
            asl_ms_log = lwcount
            cc = lwthr
            return asl_ms_log, cc
        if np.abs(lwcount - lwthr - Margin) < tol:
            asl_ms_log = lwcount
            cc =lwthr
            return asl_ms_log, cc

        midcount = (upcount + lwcount) / 2
        midthr = (upthr + lwthr) / 2
        # repeats loop until diff falls inside tolerance (-tol <= diff <= tol)
        while True:
            diff = midcount - midthr - Margin
            if np.abs(diff) <= tol:
                break
            # if tol is not met up to 20 iters, then relax tol by 10%
            iterno += 1
            if iterno > 20:
                tol *= 1.1

            if diff > tol:
                midcount = (upcount + midcount) / 2
                # upper and mid activities
                midthr = (upthr + midthr) / 2
                # ... and thresholds
            elif diff < -tol:
                # then new bounds are...
                midcount = (midcount - lwcount) / 2
                # middle and lower activities
                midthr = (midthr + lwthr) / 2
                # ... and thresholds
        # since tolerance has been satisfied, midcount is selected as
        # interpolated value with tol [dB] tolerance
        asl_ms_log = midcount
        cc = midthr
        return asl_ms_log, cc

def eval_composite(clean_utt, Genh_utt, noisy_utt=None):
    clean_utt = clean_utt.reshape(-1)
    Genh_utt = Genh_utt.reshape(-1)
    csig, cbak, covl, pesq, ssnr = CompositeEval(clean_utt,
                                                 Genh_utt,
                                                 True)
    evals = {'csig':csig, 'cbak':cbak, 'covl':covl,
             'pesq':pesq, 'ssnr':ssnr}
    if noisy_utt is not None:
        noisy_utt = noisy_utt.reshape(-1)
        csig, cbak, covl, \
        pesq, ssnr = CompositeEval(clean_utt,
                                   noisy_utt,
                                   True)
        return evals, {'csig':csig, 'cbak':cbak, 'covl':covl,
                       'pesq':pesq, 'ssnr':ssnr}
    else:
        return evals

def PESQ(ref_wav, deg_wav):
    # reference wav
    # degraded wav

    tfl = tempfile.NamedTemporaryFile()
    ref_tfl = tfl.name + '_ref.wav'
    deg_tfl = tfl.name + '_deg.wav'

    #if ref_wav.max() <= 1:
    #    ref_wav = np.array(denormalize_wave_minmax(ref_wav), dtype=np.int16)
    #if deg_wav.max() <= 1:
    #    deg_wav = np.array(denormalize_wave_minmax(deg_wav), dtype=np.int16)
	
    #wavfile.write(ref_tfl, 16000, ref_wav)
    #wavfile.write(deg_tfl, 16000, deg_wav)
    sf.write(ref_tfl, ref_wav, 16000, subtype='PCM_16')
    sf.write(deg_tfl, deg_wav, 16000, subtype='PCM_16')
    
    curr_dir = os.getcwd()
    # Write both to tmp files and then eval with pesqmain
    try:
        p = run(['pesqmain'.format(curr_dir), 
                 ref_tfl, deg_tfl, '+16000', '+wb'],
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
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    
    # scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) +
                                                        10e-20))

    # global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR

    num_frames = int(clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return overall_snr, segmental_snr

def CompositeEval(ref_wav, deg_wav, log_all=False):
    # returns [sig, bak, ovl]
    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    ref_len = ref_wav.shape[0]
    deg_wav = deg_wav[:len_]

    # Compute WSS measure
    wss_dist_vec = wss(ref_wav, deg_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(ref_wav, deg_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(ref_wav, deg_wav, 16000)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = PESQ(ref_wav, deg_wav)
    if 'error!' not in pesq_raw:
        pesq_raw = float(pesq_raw)
    else:
        pesq_raw = -1.

    def trim_mos(val):
        return min(max(val, 1), 5)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = trim_mos(Csig)
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)
    if log_all:
        return Csig, Cbak, Covl, pesq_raw, segSNR
    else:
        return Csig, Cbak, Covl

def wss(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25 # num of critical bands

    USE_FFT_SPECTRUM = 1
    n_fft = int(2 ** np.ceil(np.log(2*winlength)/np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1

    # Critical band filter definitions (Center frequency and BW in Hz)

    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30, 
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                 95.3398, 105.411, 116.256, 127.914, 140.423, 
                 153.823, 168.154, 183.457, 199.776, 217.153, 
                 235.631, 255.255, 276.072, 298.126, 321.465,
                 346.136]

    bw_min = bandwidth[0] # min critical bandwidth

    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.

    min_factor = np.exp(-30. / (2 * 2.303)) # -30 dB point of filter

    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0.append(np.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + \
                                   norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > \
                                                 min_factor)
    # For each frame of input speech, compute Weighted Spectral Slope Measure

    # num of frames
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0 # starting sample
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):

        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compuet Power Spectrum of clean and processed

        clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
        processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit
        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * \
                                     crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * \
                                         crit_filter[i, :])
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))
        # (4) Compute Spectral Shape (dB[i+1] - dB[i])

        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
        processed_slope = processed_energy[1:num_crit] - \
                processed_energy[:num_crit-1]
        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])
        # (6) Compuet the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)
        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral 
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - \
                                   clean_energy[:num_crit-1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - \
                                processed_energy[:num_crit-1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - \
                                      processed_energy[:num_crit-1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - \
                                     processed_slope[:num_crit - 1]) ** 2))

        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion

def llr(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio

    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):

        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]
        #print('A_clean shape: ', A_clean.shape)
        #print('toe(R_clean) shape: ', toeplitz(R_clean).shape)
        #print('A_clean: ', A_clean)
        #print('A_processed: ', A_processed)
        #print('toe(R_clean): ', toeplitz(R_clean))
        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        #print('num_1: {}'.format(A_processed.dot(toeplitz(R_clean))))
        #print('num: ', numerator)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
        #print('den: ', denominator)
        #log_ = np.log(max(numerator / denominator, 10e-20))
        #print('R_clean: ', R_clean)
        #print('num: ', numerator)
        #print('den: ', denominator)
        #raise NotImplementedError
        log_ = np.log(numerator / denominator)
        #print('np.log({}/{}) = {}'.format(numerator, denominator, log_))
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.array(distortion)

#@nb.jit('UniTuple(float32[:], 3)(float32[:])')#,nopython=True)
def lpcoeff(speech_frame, model_order):
    
    # (1) Compute Autocor lags
    # max?
    winlength = speech_frame.shape[0]
    R = []
    #R = [0] * (model_order + 1)
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        #raise NotImplementedError
        R.append(np.sum(first * second))
        #R[k] = np.sum( first * second)
    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        #print('-' * 40)
        #print('i: ', i)
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            #print('R[i:0:-1] = ', R[i:0:-1])
            #print('a_past = ', a_past)
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
            #print('a_past size: ', a_past.shape)
        #print('sum_term = {:.6f}'.format(sum_term))
        #print('E[i] =  {}'.format(E[i]))
        #print('R[i+1] = ', R[i+1])
        rcoeff[i] = (R[i+1] - sum_term)/E[i]
        #print('len(a) = ', len(a))
        #print('len(rcoeff) = ', len(rcoeff))
        #print('a[{}]={}'.format(i, a[i]))
        #print('rcoeff[{}]={}'.format(i, rcoeff[i]))
        a[i] = rcoeff[i]
        if i > 0:
            #print('a: ', a)
            #print('a_past: ', a_past)
            #print('a_past[:i] ', a_past[:i])
            #print('a_past[::-1] ', a_past[::-1])
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i+1] = (1-rcoeff[i]*rcoeff[i])*E[i]
        #print('E[i+1]= ', E[i+1])
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr =np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)
    #print('acorr shape: ', acorr.shape)
    #print('refcoeff shape: ', refcoeff.shape)
    #print('lpparams shape: ', lpparams.shape)
    return acorr, refcoeff, lpparams

