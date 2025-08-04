import numpy as np
import matplotlib.pyplot as plt
import colorednoise as cn

from os import makedirs
from os.path import exists, join
from scipy.signal import welch

from lagged_autocoherence import generate_surrogate, lagged_hilbert_autocoherence
from utils import scale_noise


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