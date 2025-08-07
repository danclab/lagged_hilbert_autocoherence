from lagged_autocoherence import *
import colorednoise as cn
import scipy
from scipy.optimize import curve_fit
import pandas as pd
from utils import scale_noise, sigmoid


def gen_signal_bursts(T, trials, srate, f, snr_db, num_bursts, burst_duration):
    time = np.linspace(0, T, T * srate)
    signal = np.zeros((trials, len(time)))
    total_burst_duration = num_bursts * burst_duration

    # Check if total burst duration is less than 90% of T
    if np.any(total_burst_duration > 0.9 * T):
        raise ValueError("Total burst duration exceeds 90% of signal duration.")

    # Calculate spacing between bursts
    spacing = np.zeros(num_bursts.shape)
    spacing[num_bursts > 1] = (T - total_burst_duration[num_bursts > 1]) / (num_bursts[num_bursts > 1])

    for trial in range(trials):
        # Generate burst start times without overlap
        burst_starts = np.array(
            [(i + 1) * (burst_duration[trial] + spacing[trial]) for i in range(int(num_bursts[trial]))])
        if num_bursts[trial] == 1:
            burst_starts = np.array([T / 2])

        for start in burst_starts:
            start_idx = int(start * srate)
            end_idx = start_idx + int(burst_duration[trial] * srate)
            signal_time = time[start_idx:end_idx] - start  # Resetting time for each burst
            signal[trial, start_idx:end_idx] = np.sin(2 * np.pi * f * (signal_time + np.random.randn()))

        # Add noise
        noise = cn.powerlaw_psd_gaussian(1, len(time))
        scaled_noise = scale_noise(signal[trial], noise, snr_db)
        signal[trial, :] += scaled_noise

    return signal


# Trial duration (s)
T = 10
# Number of trials
n_trials = 400
# Sampling rate
srate = 500

# Lags to evaluate at (cycles)
lags = np.arange(0.25, 20.25, .25)
# Burst frequencies (Hz)
brst_f = np.arange(20, 55, 5)

# Min and max frequencies to evaluate at
f_min = 5
f_max = 100
n_freqs = ((f_max - f_min) * 2) + 1

# SNR levels to simulate
snrs = [-50, -20, -15, -5, 0]

lfc = np.zeros((len(snrs), len(brst_f), n_trials, n_freqs, len(lags))) * np.nan
lhc = np.zeros((len(snrs), len(brst_f), n_trials, n_freqs, len(lags))) * np.nan
psds = np.zeros((len(snrs), len(brst_f), n_trials, n_freqs)) * np.nan

brst_n = np.zeros((len(snrs), len(brst_f), n_trials)) * np.nan
brst_d = np.zeros((len(snrs), len(brst_f), n_trials)) * np.nan
brst_d_c = np.zeros((len(snrs), len(brst_f), n_trials)) * np.nan

for snr_idx, snr in enumerate(snrs):
    print(f'SNR={snr} dB')
    for f_idx, f in enumerate(brst_f):
        n = np.random.randint(1, 6, size=n_trials)
        d_c = 3 * np.ones(n_trials)
        d = d_c / f

        brst_n[snr_idx, f_idx, :] = n
        brst_d_c[snr_idx, f_idx, :] = d_c
        brst_d[snr_idx, f_idx, :] = d

        print('{}Hz'.format(f))
        signal = gen_signal_bursts(T, n_trials, srate, f, snr, n, d)

        freqs, psd = scipy.signal.welch(signal, fs=srate, window='hann',
                                        nperseg=srate, noverlap=int(srate / 2), nfft=srate * 2, detrend='constant',
                                        return_onesided=True, scaling='density', axis=- 1, average='mean')
        idx = (freqs >= f_min) & (freqs <= f_max)
        freqs = freqs[idx]
        psd = psd[:, idx]
        psds[snr_idx, f_idx, :, :] = psd

        lfc[snr_idx, f_idx, :, :, :] = lagged_fourier_autocoherence(signal, freqs, lags, srate)

        lhc[snr_idx, f_idx, :, :, :] = lagged_hilbert_autocoherence(signal, freqs, lags, srate, n_jobs=-1)

np.savez('../output/sims/burst_num/sim_results',
         snrs=snrs,
         lags=lags,
         freqs=freqs,
         lfc=lfc,
         lhc=lhc,
         psds=psds,
         brst_f=brst_f,
         brst_n=brst_n,
         brst_d_c=brst_d_c,
         brst_d=brst_d)

df_snrs = np.zeros(len(snrs)*len(brst_f)*n_trials)
df_freq = np.zeros(len(snrs)*len(brst_f)*n_trials)
df_trial = np.zeros(len(snrs)*len(brst_f)*n_trials)
n_bursts = np.zeros(len(snrs)*len(brst_f)*n_trials)
x0s = np.zeros(len(snrs)*len(brst_f)*n_trials)
ks = np.zeros(len(snrs)*len(brst_f)*n_trials)

idx=0
for snr_idx in range(len(snrs)):
    for f_idx in range(len(brst_f)):
        f = brst_f[f_idx]
        lc_f_idx = np.argmin(np.abs(freqs - f))

        for i in range(n_trials):
            df_snrs[idx] = snrs[snr_idx]
            df_freq[idx] = brst_f[f_idx]
            df_trial[idx] = i + 1

            t_lhc = lhc[snr_idx, f_idx, i, :, :]
            n_bursts[idx] = brst_n[snr_idx, f_idx, i]

            b_lhc = t_lhc[lc_f_idx, :]
            p0 = [np.median(lags), 3]
            popt, pcov = curve_fit(sigmoid, lags, b_lhc, p0, method='lm', maxfev=100000)
            x0s[idx] = popt[0]
            ks[idx] = popt[1]
            idx+=1

df_burst = pd.DataFrame({
    'snr': df_snrs,
    'frequency': df_freq,
    'trial': df_trial,
    'burst_n': n_bursts,
    'x0': x0s,
    'k': ks
})

# Save to CSV
output_path_burst = '../output/sim_burst_number.csv'
df_burst.to_csv(output_path_burst, index=False)