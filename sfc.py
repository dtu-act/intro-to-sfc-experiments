import warnings
from pathlib import Path

import matplotlib as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal.windows import flattop, hann, get_window


def amplitude_spectrum(x, axis=-1):
    """Compute the amplitude spectrum of a time signal.

    Parameters
    ----------
    x : np.ndarray
        Real signal, which can be multidimensional (see axis).
    axis : int, optional
        Transformation is done along this axis. Default is -1 (last axis).

    Returns
    -------
    X: np.ndarray
        Single sided amplitude spectrum with `X.shape[axis] == x.shape[axis] // 2 + 1`.

    Notes
    -----
    If `n=len(x[axis])` is even, `x[-1]` contains the term representing both
    positive and negative Nyquist frequency (+sr/2 and -sr/2), and must also be
    purely real. If `n` is odd, there is no bin for frequency sr/2; `x[-1]`
    contains the largest positive frequency (sr/2*(n-1)/n), and is complex in
    the general case.

    """
    # move time axis to front
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]
    X = np.fft.rfft(x, axis=0) / n
    # sum complex and real part
    if n % 2 == 0:
        # zero and nyquist element only appear once in complex spectrum
        X[1:-1] *= 2
    else:
        # there is no nyquist element
        X[1:] *= 2
    # move frequency axis back again
    X = np.moveaxis(X, 0, axis)
    return X


def record_calibration_signal(
    path, channel, duration=5, sr=48000, device=None, plot=False
):
    audiodata = sd.rec(
        duration * sr,
        sr=sr,
        mapping=channel,
        blocking=True,
        device=device,
    )
    _check_audio_flags()

    sf.write(Path(path) / f"channel {channel}.wav", audiodata, sr, "FLOAT")

    if plot:
        plt.figure()
        t = time_vector(audiodata.shape[1], sr)
        plt.plot(t, audiodata.T)

    return audiodata


def calibration_gain_from_recording(file, target_level=94, plot=False):
    audiodata, sr = sf.read(file)
    num_samples = len(audiodata)

    # flattop window the recording
    window = flattop(num_samples)
    audiodata *= window / window.mean()

    target_pressure = 10 ** (target_level / 20) * 20e-6 * np.sqrt(2)
    A = amplitude_spectrum(audiodata)
    measured_pressure = np.abs(A).max()

    calibration_gain = target_pressure / measured_pressure

    if plot:
        frequencies = frequency_vector(num_samples, sr)
        plt.figure()
        A_calibrated = A * calibration_gain
        plt.plot(
            frequencies,
            20 * np.log10(np.abs(A) / np.sqrt(2) / 20e-6),
            label="uncalibrated",
        )
        plt.plot(
            frequencies,
            20 * np.log10(np.abs(A_calibrated) / np.sqrt(2) / 20e-6),
            label="calibrated",
        )
        plt.hlines(target_level, 0, sr / 2, label="94dB")
        plt.legend()
        plt.xlim(995, 1005)
        plt.ylim(90, 96)
        plt.grid(True)

    return calibration_gain


def exponential_sweep(T, sr, fade=0, f_start=None, f_end=None, pre_silence=0, post_silence=0):
    """Generate exponential sweep.

    Sweep constructed in time domain as described by `Farina`_ plus windowing.

    Parameters
    ----------
    T : float
        length of sweep
    sr : int
        sampling rate
    fade : float
        Fade in and out time in seconds.
    fstart, fstop : float or None
        Start and end frequency of sweep. If `None`, these correspond to
        `~0` and `sr/2.`
    post_silence : float
        Added zeros in seconds.

    Returns
    -------
    np.ndarray [shape=(round(T*fs),)]
        An exponential sweep

    .. _Farina:
       A. Farina, “Simultaneous measurement of impulse response and distortion
       with a swept-sine techniqueMinnaar, Pauli,” in Proc. AES 108th conv,
       Paris, France, 2000, pp. 1–15.

    """
    # TODO: use scipy.signal.chirp instead?
    n_tap = int(np.round(T * sr))

    # start and stop frequencies
    if f_start is None:
        f_start = sr / n_tap
    if f_end is None:
        f_end = sr / 2
    # validate arguments
    assert 0 < f_start < f_end
    assert f_end <= sr / 2
    # angular frequency
    omega_start = 2 * np.pi * f_start
    omega_end = 2 * np.pi * f_end
    # constuct sweep in time domain
    t = np.linspace(0, T, n_tap, endpoint=False)
    sweep = np.sin(
        omega_start
        * T
        / np.log(omega_end / omega_start)
        * (np.exp(t / T * np.log(omega_end / omega_start)) - 1)
    )
    if fade:
        n_fade = round(fade * sr)
        fading_window = hann(2 * n_fade)
        sweep[:n_fade] = sweep[:n_fade] * fading_window[:n_fade]
        sweep[-n_fade:] = sweep[-n_fade:] * fading_window[-n_fade:]
    if pre_silence > 0:
        silence = np.zeros(int(round(pre_silence * sr)))
        sweep = np.concatenate((silence, sweep))
    if post_silence > 0:
        silence = np.zeros(int(round(post_silence * sr)))
        sweep = np.concatenate((sweep, silence))
    return sweep


def multichannel_signal(x, n_ch, n_reps=1, add_reference=False):
    """Generate a serial multichannel excitation from single channel excitation.

    Parameters
    ----------
    x : np.ndarray [shape=(N,)]
        Signal to be played through each channel in series
    n_reps : int
        Number of repetitions
    n_ch : int
        Number of output channels
    reference : bool, optional
        If true, add signal at index -1 holding a repetition of `x` for use as
        reference signal.

    Returns
    -------
    np.ndarray [shape=(x.size, n_ch[+1])]
        Multichannel x.

    """
    assert x.ndim == 1
    N = x.size
    repsound = np.tile(x, n_reps)
    multisound = np.zeros((n_ch * n_reps * N, n_ch))
    for ch in range(n_ch):
        multisound[ch * n_reps * N : (ch + 1) * n_reps * N, ch] = repsound
    if add_reference:
        multisound = np.concatenate(
            (multisound, multisound.sum(axis=-1)[:, None]), axis=-1
        )
    return multisound


def regularization_fill_up_below_dynamic_range(dynamic_range_dB, x):
    """Crude choice of regularization: whenever X < max(X^2) - dynamic_range, 
    choose reg such that reg+|X|^2 == max(X^2) - dynamic_range.
    """
    X = np.fft.rfft(x)
    # maximum of reference
    maxdB = np.max(20 * np.log10(np.abs(X)))
    # power in reference should be at least
    mindB = maxdB - dynamic_range_dB
    # 10 * log10(reg + |X|**2) = mindB
    reg = 10 ** (mindB / 10) - np.abs(X) ** 2
    reg[reg < 0] = 0
    return reg


def estimate_noise_to_signal_ratio(x, y):
    """Noise to signal ratio estimate.
    
    Model: y = hx + n
           n ~ N(0, noise_power)
    """
    X = np.fft.rfft(x, axis=0)
    Y = np.fft.rfft(y, axis=0)
    noise_power = (Y.std(axis=-1)**2).reshape(Y.shape[0], -1).mean(axis=-1)
    signal_power = (np.abs(X)**2).reshape(X.shape[0], -1).mean(axis=-1)
    return noise_power / signal_power


def transfer_function(
    x, y, reg=np.finfo(float).eps, axis=0, return_time=True,
):
    """Compute FIR transfer-function between time domain signals.

    Parameters
    ----------
    x : ndarray, float
        Reference signal.
    y : ndarray, float
        Measured signal.
    reg : float or ndarray
        Regularization parameter of deconvolution. Optimally set to
        noise-to-signal ratio.
    axis : integer, optional
        Time axis of `x` and `y` over which the DFT will be applied.
    return_time : bool, optional
        If `True`, return impulse response. Otherwise, return frequency response.

    Notes
    -----
    For multichannel data given as ndarrays, `np.fft.rfft(x)`, `np.fft.rfft(x)`
    and `reg` must broadcast.

    Returns
    -------
    h : ndarray, float
        Impulse response (if `return_time==True`) or frequency response (else) of system
        between `x` and `y`.

    """
    n = x.shape[axis]
    # move time axis to last dimension for easy broadcasting
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    # FFT
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    # regularized deconvolution in frequency domain
    H = Y * X.conj() / (np.abs(X) ** 2 + reg)
    # move axis back
    H = np.moveaxis(H, -1, axis)
    if return_time:
        h = np.fft.irfft(H, axis=axis, n=n)
        return h
    return H


### recording with single microphones

def _record_single_excitation(x, out_ch=1, in_ch=1, squeeze=True, **sd_kwargs):
    """Record a single (possibly multi-channel) excitation at multiple inputs.

    Parameters
    ----------
    x :  np.ndarray [shape=(nt,[n_out])]
        Excitation signal
    out_ch : int or list of length `n_out, optional
        Output channels
    in_ch : int or list of length `n_in`, optional
        Input channels
    squeeze : bool
        Remove singleton dimensions of output
    sd_kwargs
        Keyword arguments passed to `sounddevice.playrec`

    Returns
    -------
    np.ndarray [shape=(nt,[n_in])]
        Impulse response between output channel and input channels

    """
    out_ch = np.atleast_1d(out_ch)
    in_ch = np.atleast_1d(in_ch)
    data = sd.playrec(
        x, input_mapping=in_ch, output_mapping=out_ch, blocking=True, **sd_kwargs,
    )
    _check_audio_flags()
    if squeeze:
        return data.squeeze()
    return data


def measure_via_audio_interface(x, out_chs, in_ch=1, n_reps=1, squeeze=True, **sd_kwargs):
    """Make a measurement via an audio interface.

    This is a convienience wrapper around `sounddevice.playrec`.

    Parameters
    ----------
    x : np.ndarray [shape=(nt,[n_out])] or list of such arrays
        Excitation signal or list (length `n_sys`) of excitation signals
    out_chs : list of int or list of such lists
        Output channels or list (length `n_sys`) of output channels
    in_ch : int or list
        Input channels
    n_reps : int, optinal
        Number of repetitions of each x-out_chs pair
    squeeze : bool
        Remove singleton dimensions of output

    Returns
    -------
    np.ndarrat [shape=(nt, n_in, n_sys, n_reps)]
        Recorded signals. If `squeeze == True`, axis of dimension 1 are removed
        from the ouput.

    Notes
    -----
    If both `x` and `out_chs` are lists, they must be of the same length. If
    `out_chs` is a list of `int`, the same outputs are used for each signal in
    x. If `x` is a single ndarray, it is used for each output set specified in
    out_chs.

    """
    in_ch = np.atleast_1d(in_ch)

    if isinstance(x, list) and (isinstance(out_chs, list) and isinstance(out_chs[0], list)):
        # multiple outputs, each with their own excitation signal
        assert len(x) == len(out_chs)
        n_sys = len(out_chs)
    elif isinstance(x, list) and isinstance(out_chs[0], int):
        # multiple signals over the same output
        n_sys = len(x)
        out_chs = np.broadcast_to(out_chs, (n_sys, len(out_chs)))
    elif isinstance(x, np.ndarray) and isinstance(out_chs, list) and isinstance(out_chs[0], list):
        # same signal over multiple outputs
        n_sys = len(out_chs)
        x = np.broadcast_to(x, (n_sys,) + x.shape)
    elif isinstance(x, np.ndarray) and isinstance(out_chs, list) and isinstance(out_chs[0], int):
        # same signal over one set of outputs
        n_sys = 1
        x = [x]
    else:
        raise ValueError('Invalid combination of `x` and `out_chs`')

    data = np.zeros((x[0].shape[0], len(in_ch), n_sys, n_reps))
    for i in range(n_sys):
        for j in range(n_reps):
            data[..., i, j] = _record_single_excitation(
                x[i], out_ch=out_chs[i], in_ch=in_ch, squeeze=False, **sd_kwargs
            )
    if squeeze:
        return data.squeeze()
    return data

### recording with external front end

def load_bk_wav_recording(file, n_ch=1, n_reps=1, has_reference=True):
    """Load multichannel B&K Time Data Recording saved as WAV file.

    Cuts recording into chunks of equal length according to the number of
    output channel sets and repetitions.

    Parameters
    ----------
    file : str or file_like
        path to audio file
    n_ch : int, optional
        Number of in series recorded output channels.
    n_reps : int, optional
        Number of recorded repetitions.

    Returns
    -------
    x : np.ndarray [shape=(n_samp, 1, n_ch, n_reps)]
        Reference signal. Only returned if `has_reference==True`.
    y : np.ndarray [shape=(n_samp, n_in, n_ch, n_reps)]
        Microphone signals.
    sr : int
        Samplerate.

    """
    data, sr = sf.read(file)
    if data.shape[0] % (n_ch * n_reps) != 0:
        # remove samples at the end if not divisable by `n_ch * n_reps`
        data = data[:- (data.shape[0] % (n_ch * n_reps))]
    # forcing into new shape / cutting recording into sections
    data = np.stack(np.split(np.stack(
        np.split(data, n_ch, axis=0), axis=-1), n_reps, axis=0), axis=-1)
    if has_reference:
        return data[:, :1], data[:, 1:], sr
    return data, sr


def bk_planar_array_mic_positions():
    """Cartesian coordinates of ACT's planar array microphones.

    Returns
    -------
    r : np.ndarray(shape=[60, 2])

    """
    x = np.linspace(0, 0.675, 10)
    y = np.linspace(0.375, 0, 6)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.array((X.flatten(), Y.flatten()))
    return r

### Filter computation

def pressure_matching(H, h_target, reg=np.finfo(float).eps):
    """Compute loudspeaker weights via pressure matching with l2 regularization.

    Solves the following least-square problem indepedently for each frequency:

        min_w  ||H*w - h_target||² + reg*||w||²

    Parameters
    ----------
    H : np.ndarray[shape=(nf, nm, ns), dtype=complex]
    h_target : np.ndarray[shape=(nf, nm), dtype=complex]
    reg : float or np.ndarray[shape=(nf,)]
        Regularization Parameter

    Returns
    -------
    w : np.ndarray[shape=(nf, ns), dtype=complex]
        Complex weights

    """
    nf, _, ns = H.shape

    reg = np.broadcast_to(reg, (nf,))

    w = np.zeros((nf, ns), dtype=complex)
    for i in range(nf):
        # Solve equivalent least-squares problem
        #
        # min_w ||[[H            ],         [[h_target], ||^2
        #       ||[sqrt(reg) * I]]  *  w -  [0       ]]  ||
        #
        # NOTE: one could also use sklearn.linear_model.Ridge
        A = np.concatenate((H[i], np.sqrt(reg[i]) * np.identity(ns)), axis=0)
        b = np.concatenate((h_target[i], np.zeros(ns)), axis=0)
        w[i] = np.linalg.lstsq(A, b, rcond=None)[0]

    return w

#### utils

def _check_audio_flags():
    status = sd.get_status()
    if status.input_underflow:
        warnings.warn("Input underflow")
    if status.input_overflow:
        warnings.warn("Input overflow")
    if status.output_overflow:
        warnings.warn("output overflow")
    if status.output_underflow:
        warnings.warn("output underflow")
    if status.priming_output:
        warnings.warn("Primed output")


def _sample_window(n, startwindow, stopwindow, window):
    """Create a sample domain window."""
    swindow = np.ones(n)

    if startwindow is not None:
        length = startwindow[1] - startwindow[0]
        w = get_window(window, 2 * length, fftbins=False)[:length]
        swindow[: startwindow[0]] = 0
        swindow[startwindow[0] : startwindow[1]] = w

    if stopwindow is not None:
        # stop window
        length = stopwindow[1] - stopwindow[0]
        w = get_window(window, 2 * length, fftbins=False)[length:]
        swindow[stopwindow[0] + 1 : stopwindow[1] + 1] = w
        swindow[stopwindow[1] + 1 :] = 0

    return swindow


def time_window(n, sr, startwindow, stopwindow, window="hann"):
    """Create a time domain window.

    If `startwindow` or `stopwindow` are given as tuples (e.g.
    `(start, stop)`), they define the intervals over which the window opens or
    closes. If either is `None`, the window starts/stays open from/to the
    start/end.

    The times `start` and `stop` can be negative in which case they define times
    relative to the end of the response. If `stop` is None, it denotes the end
    of the response.

    """
    times = time_vector(n, sr)
    T = times[-1] + times[1]  # total time length

    if startwindow is None:
        startwindow_n = None
    else:
        startwindow_n = []
        for t in startwindow:
            if t < 0:
                t += T
            assert 0 <= t or t <= T
            startwindow_n.append(_find_nearest(times, t)[1])

    if stopwindow is None:
        stopwindow_n = None
    else:
        stopwindow_n = []
        for t in stopwindow:
            if t is None:
                t = times[-1]
            elif t < 0:
                t += T
            assert 0 <= t or t <= T
            stopwindow_n.append(_find_nearest(times, t)[1])

    twindow = _sample_window(n, startwindow_n, stopwindow_n, window)

    return twindow


def frequency_window(n, sr, startwindow, stopwindow, window="hann"):
    """Create a frequency domain window.

    Works similar to `time_window`.
    """
    freqs = frequency_vector(n, sr)

    if startwindow is None:
        startwindow_n = None
    else:
        startwindow_n = [_find_nearest(freqs, f)[1] for f in startwindow]

    if stopwindow is None:
        stopwindow_n = None
    else:
        stopwindow_n = [_find_nearest(freqs, f)[1] for f in stopwindow]

    fwindow = _sample_window(len(freqs), startwindow_n, stopwindow_n, window)

    return fwindow


def time_window_around_peak(ir, sr, tleft, tright, window='tukey', param=0.5):
    """Create time window around maximum of response.

    Parameters
    ----------
    ir : array_like
        Input response with time axis at first dimension
    sr : int
        Sample rate
    tleft : float
        Start of time window relative to impulse response peak in seconds
    tright : float
        End of time window relative to impulse response peak in seconds
    window : str
        Window type for `scipy.signal.get_window`
    param : float or None, optional
        Specify parameter for some windows, e.g. for `tukey` parameter
        corresponds to `alpha`.

    Returns
    -------
    np.ndarray[shape=ir.shape]
        Time windows.
    """
    orig_shape = ir.shape
    ir = ir.reshape(ir.shape[0], -1)
    # convert time to samples
    samples_left = int(sr * tleft)
    samples_right = int(sr * tright)
    # construct window
    windows = np.ones(ir.shape)
    for i in range(ir.shape[1]):
        idx_peak = np.argmax(np.abs(ir[:, i]))
        idx_window_start = max(idx_peak - samples_left, 0)
        idx_window_end = min(idx_peak + samples_right, ir.shape[0])
        if param is not None:
            w = get_window((window, param), idx_window_end - idx_window_start)
        else:
            w = get_window(window, idx_window_end - idx_window_start)
        windows[idx_window_start:idx_window_end, i] *= w
        windows[:idx_window_start, i] = 0
        windows[idx_window_end:, i] = 0
    return windows.reshape(orig_shape)


def time_vector(n, sr):
    """Time values of filter with n taps sampled at sr.

    Parameters
    ----------
    n : int
        Number of taps
    sr : int
        Samplerate

    Returns
    -------
    np.ndarray [shape=(n,)]
        Times in seconds

    """
    T = 1 / sr
    return np.arange(n, dtype=float) * T  # use float against int wrapping


def frequency_vector(n, sr, sided="single"):
    """Frequency values of filter with n taps sampled at sr up to Nyquist.

    Parameters
    ----------
    n : int
        Number of taps
    sr : int
        Samplerate
    sided: str
        Generate frequencies for a "single" or "double" sided spectrum

    Returns
    -------
    np.ndarray [shape=(n // 2 + 1,) or (n,)]
        Frequencies in Hz

    """
    # use float to protect against int wrapping
    if sided == "single":
        f = np.arange(n // 2 + 1, dtype=float) * sr / n
    elif sided == "double":
        f = np.arange(n, dtype=float) * sr / n
    else:
        raise ValueError("Invalid value for `sided`.")

    return f


def _find_nearest(array, value):
    """Find nearest value in an array and its index."""
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def olafilt(b, x, subscripts=None, zi=None):
    """Filter a multi dimensional array with an FIR filter matrix.

    Filter a data sequence, `x`, using a FIR filter given in `b`. Filtering uses
    the overlap-add method converting both `x` and `b` into frequency domain
    first.  The FFT size is determined as the next higher power of 2 of twice
    the length of `b`. Multi-channel fitering is support via `numpy.einsum`
    notation.

    Parameters
    ----------
    b : array_like, shape (m[, ...])
        Filter matrix with `m` taps.
    x : array_like, shape (n[, ...])
        Input signal.
    subscripts : str or None, optional
        String that defines the matrix operations in the multichannel case using
        the notation from `numpy.einsum`. Subscripts for `b` and `x` and output
        must start with the same letter, e.g. `nlmk,nk->nl`.
    zi : int or array_like, shape (m - 1[, ...]), optional
        Initial condition of the filter, but in reality just the runout of the
        previous computation.  If `zi` is None (default), then zero initial
        state is assumed. Zero initial state can be explicitly passes with `0`.
        Shape after first dimention must be compatible with output defined via
        `subscripts`.

    Returns
    -------
    y : numpy.ndarray
        The output of the digital filter. The precise output shape is defined by
        `subscripts`, but always `y.shape[0] == n`.
    zf : numpy.ndarray
        If `zi` is None, this is not returned, otherwise, `zf` holds the final
        filter state. The precise output shape is defined by `subscripts`, but
        always `zf.shape[0] == m - 1`.

    Notes
    -----
    Taken from `https://github.com/fhchl/adafilt`

    """
    b = np.asarray(b)
    x = np.asarray(x)

    if (b.ndim > 1 or x.ndim > 1) and subscripts is None:
        raise ValueError("Supply `subscripts` argument for multi-channel filtering.")

    L_I = b.shape[0]
    L_sig = x.shape[0]

    # find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    L_F = int(2 << (L_I - 1).bit_length())  # FFT Size
    L_S = L_F - L_I + 1  # length of segments
    offsets = range(0, L_sig, L_S)

    if subscripts is None:
        outshape = L_sig + L_F
    else:
        outshape = (L_sig + L_F, *_einsum_outshape(subscripts, b, x)[1:])

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        res = np.zeros(outshape, dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        res = np.zeros(outshape)

    B = fft_func(b, n=L_F, axis=0)

    # overlap and add
    for n in offsets:
        Xseg = fft_func(x[n : n + L_S], n=L_F, axis=0)

        if subscripts is None:
            # fast 1D case
            C = B * Xseg
        else:
            C = np.einsum(subscripts, B, Xseg)

        res[n : n + L_F] += ifft_func(C, axis=0)

    if zi is not None:
        res[: L_I - 1] = res[: L_I - 1] + zi
        return res[:L_sig], res[L_sig : L_sig + L_I - 1]

    return res[:L_sig]


def _einsum_outshape(subscripts, *operants):
    """Compute the shape of output from `numpy.einsum`.

    Does not support ellipses.
    """
    if "." in subscripts:
        raise ValueError(f"Ellipses are not supported: {subscripts}")

    insubs, outsubs = subscripts.replace(",", "").split("->")
    if outsubs == "":
        return ()
    insubs = np.array(list(insubs))
    innumber = np.concatenate([op.shape for op in operants])
    outshape = []
    for o in outsubs:
        indices, = np.where(insubs == o)
        try:
            outshape.append(innumber[indices].max())
        except ValueError:
            raise ValueError(f"Invalid subscripts: {subscripts}")
    return tuple(outshape)