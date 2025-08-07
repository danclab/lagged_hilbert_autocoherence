import math
import warnings

import numpy as np
import colorednoise as cn
import statsmodels.api as sm
from joblib import Parallel, delayed
from mne.filter import filter_data
from scipy.signal import welch, hilbert

from utils import scale_noise

def generate_surrogate(signal, n_shuffles=1000, method="phase", n_jobs=-1):
    """
    Generate per-trial surrogate amplitude products via ARMA or phase randomization.

    Parameters
    ----------
    signal : ndarray
        Input signal, shape (n_trials, n_pts).
    n_shuffles : int
        Number of surrogate realizations per trial.
    method : str
        'arma' or 'phase'.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    amp_prod : ndarray
        Shape: (n_trials, n_shuffles, n_pts - 1)
    """

    signal = np.atleast_2d(signal)
    n_trials, n_pts = signal.shape

    def generate_surrogate_trace(trial):
        if method == "arma":
            mdl = sm.tsa.ARIMA(trial, order=(1, 0, 0))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = mdl.fit()
                ar_process = sm.tsa.ArmaProcess.from_estimation(result)
                return ar_process.generate_sample(nsample=n_pts, scale=result.resid.std())

        elif method == "phase":
            fft = np.fft.rfft(trial)
            amp = np.abs(fft)
            rand_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=fft.shape))
            return np.fft.irfft(amp * rand_phase, n=n_pts)

        else:
            raise ValueError(f"Unknown method: {method}")

    def sim_per_trial(trial_idx):
        trial = signal[trial_idx]
        return np.stack([
            np.abs(hilbert(s := generate_surrogate_trace(trial)))[:-1] *
            np.abs(hilbert(s))[1:]
            for _ in range(n_shuffles)
        ], axis=0)  # shape: (n_shuffles, n_pts - 1)

    amp_prod = Parallel(n_jobs=n_jobs)(
        delayed(sim_per_trial)(i) for i in range(n_trials)
    )

    return np.stack(amp_prod, axis=0)  # shape: (n_trials, n_shuffles, n_pts - 1)



def lagged_hilbert_autocoherence(signal, freqs, lags, srate, df=None, n_shuffles=1000, type='coh',
                                 thresh_prctile=95, surr_method="phase", surr_data=None, n_jobs=-1):
    """
    Compute lagged Hilbert autocoherence (or phase-locking value or amplitude autocoherence) for a signal.

    Parameters
    ----------
    signal : ndarray
        The input signal, shape (n_trials, n_pts).
    freqs : array_like
        Frequencies of interest.
    lags : array_like
        Lags of interest.
    srate : float
        Sampling rate in Hz.
    df : float
        Frequency resolution in Hz. If None (default), computed as the difference in frequencies.
    n_shuffles: integer
        Number of times to shuffle data
    type : str
        Type of output: 'coh' for lagged autocoherence, 'plv' for lagged phase-locking value, or 'amp_coh' for lagged amplitude
        autocoherence.
    thresh_prctile: integer or None
        Percentile used to compute threshold of statistical significance based on surrogate data.
        If None no surrogate data is created and no threshold is applied.
    surr_method: str (default "arma")
        Type of method used to create surrogate date:
        'arma' for signals with shuffled amplitude coefficients, or
        'phase' for signals with shuffled phase coefficients.
    surr_data: None or ndarray (default "None")
        Surrogate data are not generated if provided by the user (trials x shuffles).
    n_jobs: integer
        The number of parallel jobs to run (default = -1). -1 means using all processors.

    Returns
    -------
    lcs : ndarray
        The output, shape (n_trials, n_freqs, n_lags).
    """

    n_trials = signal.shape[0]
    n_pts = signal.shape[-1]
    T = n_pts * 1 / srate
    time = np.linspace(0, T, int(T * srate))

    # Number of frequencies
    n_freqs = len(freqs)
    # Number of lags
    n_lags = len(lags)

    # Check that epochs are long enough for requested frequencies and lags - just have to check number of eval pts for
    # lowest frequency and longest lag
    min_freq=np.min(freqs)
    max_lag=np.max(lags)
    lag_dur_s = np.max([max_lag / min_freq, 1 / srate])
    min_epoch_len = 2 * lag_dur_s
    if T<min_epoch_len:
        raise ValueError('Epoch length must be at least {min_len:.2f}s to evaluate LHC at {min_freq:.2f} Hz and {max_lag:.2f} cycles'.format(
            min_len=min_epoch_len,
            min_freq=min_freq,
            max_lag=max_lag))

    # Frequency resolution
    if df is None:
        df = np.diff(freqs)[0]

    # Compute threshold as 5th percentile of shuffled amplitude
    # products
    if thresh_prctile != None:
        if surr_data is None:
            filtered_signal = filter_data(signal, srate, freqs[0], freqs[-1], verbose=False)
            surr_data = generate_surrogate(filtered_signal, n_shuffles=n_shuffles,
                                           method=surr_method, n_jobs=n_jobs)
            # Average over time dimension
            surr_data = np.mean(surr_data, axis=-1)
        else:
            n_dim = len(surr_data.shape)
            if n_dim != 2:
                raise ValueError("Expected n dim of surrogate data is 2 (trials x shuffles), but {} provided instead.".format(n_dim))
        thresh = np.percentile(surr_data, thresh_prctile, axis=-1)
    else:
        if type == 'coh':
            type_str = "lagged Hilbert autocoherence"
        elif type == 'plv':
            type_str = "phase-locking value"
        elif type == 'amp_coh':
            type_str = "lagged amplitude autocoherence"
        print("Computing {} without surrogate analysis and thresholding.".format(type_str))

    padd_signal = np.hstack([np.zeros((n_trials, n_pts)), signal, np.zeros((n_trials, n_pts))])
    signal_fft = np.fft.rfft(padd_signal, axis=-1)
    fft_frex = np.fft.rfftfreq(padd_signal.shape[-1], d=1 / srate)
    sigma = df * .5

    def run_freq(f_idx):
        freq = freqs[f_idx]

        f_lcs = np.zeros((n_trials, n_lags))

        # Bandpass filtering using multiplication by a Gaussian kernel
        # in the frequency domain

        # Gaussian kernel centered on frequency with width defined
        # by requested frequency resolution
        kernel = np.exp(-((fft_frex - freq) ** 2 / (2.0 * sigma ** 2)))

        # Multiply Fourier-transformed signal by kernel
        fsignal_fft = np.multiply(signal_fft, kernel)
        # Reverse Fourier to get bandpass filtered signal
        f_signal = np.fft.irfft(fsignal_fft, axis=-1)

        # Get analytic signal of bandpass filtered data (phase and amplitude)
        analytic_signal = hilbert(f_signal, N=None, axis=-1)
        # Cut off padding
        analytic_signal = analytic_signal[:, len(time):2 * len(time)]

        for l_idx, lag in enumerate(lags):
            # Duration of this lag in s
            lag_dur_s = np.max([lag / freq, 1/srate])

            # Number of evaluations
            n_evals = int(np.floor(T / lag_dur_s))
            # Remaining time
            diff = T - (n_evals * lag_dur_s)

            # Start time
            start_time = time[0]
            # Evaluation times (ms)
            eval_times = np.linspace(start_time, T - diff, n_evals + 1)[:-1]
            # Evaluation time points
            eval_pts = np.searchsorted(time, eval_times)

            # Number of points between the first and next evaluation time points
            n_range = eval_pts[1] - eval_pts[0]
            # Analytic signal at n=0...n_evals-1 evaluation points, and m=0..n_range time points in between
            f1 = analytic_signal[:,eval_pts[:-1, np.newaxis] + np.arange(n_range)]
            # Analytic signal at n=1...n_evals evaluation points, and m=0..n_range time points in between
            f2 = analytic_signal[:,eval_pts[1:, np.newaxis] + np.arange(n_range)]

            # Calculate the phase difference and amplitude product
            phase_diff = np.angle(f2) - np.angle(f1)
            amp_prod = np.abs(f1) * np.abs(f2)

            if type == 'coh':
                # Lagged autocoherence
                num = np.sum(amp_prod * np.exp(complex(0, 1) * phase_diff), axis=1)
                f1_pow = np.power(f1, 2)
                f2_pow = np.power(f2, 2)
                denom = np.sqrt(np.sum(np.abs(f1_pow), axis=1) * np.sum(np.abs(f2_pow), axis=1))

                lc = np.abs(num / denom)

                if thresh_prctile != None:
                    lc[denom<np.tile(thresh, (lc.shape[1], 1)).T]=0

            elif type == 'plv':
                expected_phase_diff = lag * 2 * math.pi
                num = np.sum(np.exp(complex(0, 1) * (expected_phase_diff - phase_diff)), axis=1)
                denom = len(eval_pts) - 1

                f1_pow = np.power(f1, 2)
                f2_pow = np.power(f2, 2)
                amp_denom = np.sqrt(np.sum(np.abs(f1_pow), axis=1) * np.sum(np.abs(f2_pow), axis=1))

                lc = np.abs(num / denom)

                if thresh_prctile != None:
                    lc[denom<np.tile(thresh, (lc.shape[1], 1)).T]=0

            elif type == 'amp_coh':
                # Numerator - sum is over evaluation points
                num = np.sum(amp_prod, axis=1)
                f1_pow = np.power(f1, 2)
                f2_pow = np.power(f2, 2)
                denom = np.sqrt(np.sum(np.abs(f1_pow), axis=1) * np.sum(np.abs(f2_pow), axis=1))
                lc = num / denom

                if thresh_prctile != None:
                    lc[denom<np.tile(thresh, (lc.shape[1], 1)).T]=0

            # Average over the time points in between evaluation points
            f_lcs[:, l_idx] = np.mean(lc, axis=-1)

        return f_lcs

    lcs = Parallel(
        n_jobs=n_jobs
    )(delayed(run_freq)(f) for f in range(n_freqs))

    lcs = np.array(lcs)
    lcs = np.moveaxis(lcs, [0, 1, 2], [1, 0, 2])
    return lcs


def gen_signal(T, trials, srate, fs, snr_db):
    time = np.linspace(0, T, T * srate)

    w = 2. * np.pi * f
    signal = np.zeros((trials, len(time)))
    for i in range(trials):
        pure_signal = np.sin(w * (time + np.random.randn()))
        noise = cn.powerlaw_psd_gaussian(1, len(time))
        scaled_noise = scale_noise(pure_signal, noise, snr_db)

        signal[i, :] = pure_signal + scaled_noise

    return signal

surr_methods = ["arma", "phase"]

# Signal hyper-params.
T = 10
n_trials = 100
srate = 500
snr = -20 # 0

# Frequency resolution.
f_min = 5
f_max = 100
n_freqs = ((f_max - f_min) * 2) + 1
osc_f = np.array([10, 25, 50])

# Surrogate analysis.
lags = np.arange(0.25, 6.25, 0.25)
n_shuffles = 1000

f_psds = []
f_surrogate_psds = []
f_surrogate_lhc = []

for f_idx, f in enumerate(osc_f):

    print('Simulating {} Hz oscillatory signals.'.format(f))

    signals = gen_signal(T, n_trials, srate, f, snr)

    freqs, psds = welch(
        signals,
        fs=srate,
        window='hann',
        nperseg=srate,
        noverlap=int(srate / 2),
        nfft=srate * 2,
        detrend='constant',
        return_onesided=True,
        scaling='density',
        axis=-1,
        average='mean'
    )

    idx = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[idx]
    psds = psds[:, idx]
    f_psds.append(psds)

    m_psds = []
    m_lhc = []
    for method in surr_methods:
        surrogate_signals = generate_surrogate(
            signals,
            n_shuffles=n_shuffles,
            method=method,
            n_jobs=-1,
        )

        freqs, surrogate_psds = welch(
            surrogate_signals,
            fs=srate,
            window='hann',
            nperseg=srate,
            noverlap=int(srate / 2),
            nfft=srate * 2,
            detrend='constant',
            return_onesided=True,
            scaling='density',
            axis=-1,
            average='mean'
        )

        idx = (freqs >= f_min) & (freqs <= f_max)
        freqs = freqs[idx]
        surrogate_psds = surrogate_psds[:, :, idx]
        m_psds.append(surrogate_psds)

        lhc = lagged_hilbert_autocoherence(signals, freqs, lags, srate, surr_method=method, n_jobs=-1)
        m_lhc.append(lhc)
    f_surrogate_psds.append(m_psds)
    f_surrogate_lhc.append(m_lhc)

f_psds = np.array(f_psds)
f_surrogate_psds = np.array(f_surrogate_psds)
f_surrogate_lhc = np.array(f_surrogate_lhc)


np.savez('../output/sims/surrogate/sim_results',
         osc_f=osc_f,
         surr_methods=surr_methods,
         lags=lags,
         freqs=freqs,
         f_psds=f_psds,
         f_surrogate_psds=f_surrogate_psds,
         f_surrogate_lhc=f_surrogate_lhc)