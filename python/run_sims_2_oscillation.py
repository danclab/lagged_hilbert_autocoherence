from lagged_autocoherence import *
import colorednoise as cn
import scipy
import pandas as pd

from utils import scale_noise

def gen_signal(T, trials, srate, f, snr_db):
    time = np.linspace(0, T, T * srate)

    w = 2. * np.pi * f
    signal = np.zeros((trials, len(time)))
    for i in range(trials):
        pure_signal = np.sin(w * (time + np.random.randn()))
        noise = cn.powerlaw_psd_gaussian(1, len(time))
        scaled_noise = scale_noise(pure_signal, noise, snr_db)

        signal[i, :] = pure_signal + scaled_noise

    return signal

# Trial duration (s)
T = 5
# Number of trials
n_trials = 100
# Sampling rate
srate = 500
# Lags to evaluate at (cycles)
lags = np.arange(1, 6.5, .5)
# Oscillation frequencies (Hz)
osc_f = np.arange(10, 55, 5)

# Min and max frequencies to evaluate at
f_min = 5
f_max = 100
n_freqs = ((f_max - f_min) * 2) + 1

# SNR levels to simulate
snrs = [-50, -20, -15, -5, 0]

lfc = np.zeros((len(snrs), len(osc_f), n_trials, n_freqs, len(lags)))
lhc = np.zeros((len(snrs), len(osc_f), n_trials, n_freqs, len(lags)))
psds = np.zeros((len(snrs), len(osc_f), n_trials, n_freqs))

for snr_idx, snr in enumerate(snrs):
    print(f'SNR={snr} dB')
    for f_idx, f in enumerate(osc_f):
        print(f'{f} Hz')
        signal = gen_signal(T, n_trials, srate, f, snr)

        freqs, psd = scipy.signal.welch(signal, fs=srate, window='hann',
                                        nperseg=srate, noverlap=int(srate / 2), nfft=srate * 2, detrend='constant',
                                        return_onesided=True, scaling='density', axis=- 1, average='mean')
        idx = (freqs >= f_min) & (freqs <= f_max)
        freqs = freqs[idx]
        psd = psd[:, idx]
        psds[snr_idx, f_idx, :, :] = psd

        lfc[snr_idx, f_idx, :, :, :] = lagged_fourier_autocoherence(signal, freqs, lags, srate)

        lhc[snr_idx, f_idx, :, :, :] = lagged_hilbert_autocoherence(signal, freqs, lags, srate, n_jobs=-1)

np.savez('../output/sims/oscillation/sim_results',
         snrs=snrs,
         lags=lags,
         freqs=freqs,
         lfc=lfc,
         lhc=lhc,
         psds=psds,
         osc_f=osc_f)

df_snrs = np.zeros(len(snrs)*len(osc_f)*n_trials)
df_freq = np.zeros(len(snrs)*len(osc_f)*n_trials)
df_trial = np.zeros(len(snrs)*len(osc_f)*n_trials)
lfc_rmse = np.zeros(len(snrs)*len(osc_f)*n_trials)
lhc_rmse = np.zeros(len(snrs)*len(osc_f)*n_trials)
lfc_std = np.zeros(len(snrs)*len(osc_f)*n_trials)
lhc_std = np.zeros(len(snrs)*len(osc_f)*n_trials)

idx=0
for snr_idx in range(len(snrs)):
    for f_idx in range(len(osc_f)):
        for n in range(n_trials):
            df_snrs[idx]=snrs[snr_idx]
            df_freq[idx]=osc_f[f_idx]
            df_trial[idx]=n+1

            t_lfc = lfc[snr_idx, f_idx, n, :, :]
            t_lhc = lhc[snr_idx, f_idx, n, :, :]

            f_psd_lfc = np.nanmean(t_lfc, axis=1)
            lfc_std[idx] = np.nanstd(f_psd_lfc)
            f_psd_lfc = f_psd_lfc / np.nanmax(f_psd_lfc)

            f_psd_lhc = np.nanmean(t_lhc, axis=1)
            lhc_std[idx] = np.nanstd(f_psd_lhc)
            f_psd_lhc = f_psd_lhc / np.nanmax(f_psd_lhc)

            f_psd = psds[snr_idx, f_idx, n, :]
            f_psd = f_psd / np.max(f_psd)
            signal_power = np.nansum(f_psd)
            lfc_rmse[idx] = np.sqrt(np.nansum((f_psd_lfc - f_psd) ** 2))/signal_power
            lhc_rmse[idx] = np.sqrt(np.nansum((f_psd_lhc - f_psd) ** 2))/signal_power
            idx+=1

# Create DataFrame for LFaC
df_lfc = pd.DataFrame({
    'snr': df_snrs,
    'frequency': df_freq,
    'trial': df_trial,
    'rmse': lfc_rmse,
    'std': lfc_std,
    'algorithm': 'LFaC'
})

# Create DataFrame for LHaC
df_lhc = pd.DataFrame({
    'snr': df_snrs,
    'frequency': df_freq,
    'trial': df_trial,
    'rmse': lhc_rmse,
    'std': lhc_std,
    'algorithm': 'LHaC'
})

# Concatenate the two DataFrames
df_combined = pd.concat([df_lfc, df_lhc])

# Save to CSV
output_path = '../output/sim_oscillations.csv'
df_combined.to_csv(output_path, index=False)