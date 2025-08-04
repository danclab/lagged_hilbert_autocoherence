import numpy as np
import scipy
from scipy.signal import hilbert
from lagged_autocoherence import generate_surrogate, filter_data
from utils import scale_noise
import colorednoise as cn

def gen_signal_bursts(T, n_trials, srate, f, n_cycles, num_bursts, snr_db):
    dt=1/srate
    time = np.linspace(0, T, T * srate)
    signal=np.zeros((n_trials, len(time)))

    for trial in range(n_trials):
        # Burst signal
        s1=np.zeros((len(time)))

        # Keep track of burst start and stop times so they
        # don't overlap
        burst_starts=[]
        burst_stops=[]

        # Burst duration in seconds
        dur_s=n_cycles/f
        # Burst duration in time steps
        dur_pts=int(dur_s/dt)

        # Random start and stop time
        start=np.random.randint(0,len(time)-dur_pts)
        stop=start+dur_pts
        s1[start:stop]=np.sin(2. * np.pi * f1 * (time[start:stop]-time[start]))
        burst_starts.append(start)
        burst_stops.append(stop)

        while len(burst_starts)<num_bursts:
            # Random start and stop time
            start=np.random.randint(int(T/dt)-dur_pts)
            stop=start+dur_pts

            # Check that doesn't overlap with other bursts
            overlap=False
            for (other_start,other_stop) in zip(burst_starts,burst_stops):
                if (start >= other_start and start < other_stop) or (stop > other_start and stop <= other_stop):
                    overlap=True
                    break

            # Generate burst
            if not overlap:
                s1[start:stop]=np.sin(2. * np.pi * f1 * (time[start:stop]+np.random.randn()))
                burst_starts.append(start)
                burst_stops.append(stop)

        # Generated signal
        noise = cn.powerlaw_psd_gaussian(1, len(time))
        scaled_noise = scale_noise(s1, noise, snr_db)
        signal[trial,:]=s1+scaled_noise
    return signal


# --- Generate signal and compute LHaC ---

# Duration (s)
T = 10
# Sampling rate
srate = 500
time = np.linspace(0, T, T * srate)

# Burst frequency
f1 = 15
# Length of bursts in cycles
f1_num_cycles = 5
# Number of bursts
f1_num_bursts = 1

n_trials = 100

lags = np.arange(0.5, 10.5, 0.5)
f_min = 5
f_max = 100

snrs = [-50, -20, -15, -5, 0]

snr_thresholds = []
for snr in snrs:
    signal = gen_signal_bursts(T, n_trials, srate, f1, f1_num_cycles, f1_num_bursts, snr)

    freqs, psd = scipy.signal.welch(signal, fs=srate, window='hann',
                                    nperseg=srate, noverlap=int(srate / 2), nfft=srate * 2, detrend='constant',
                                    return_onesided=True, scaling='density', axis=- 1, average='mean')
    idx = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[idx]
    psd = psd[:, idx]

    filtered_signal = filter_data(signal, srate, freqs[0], freqs[-1], verbose=False)

    surr_data = generate_surrogate(filtered_signal, n_shuffles=1000, method='phase', n_jobs=-1)

    joint_energy_surrogates = np.mean(surr_data, axis=-1)
    thresholds = np.percentile(joint_energy_surrogates, 95, axis=-1)
    snr_thresholds.append(thresholds)
snr_thresholds = np.array(snr_thresholds)

np.savez(
    '../output/sims/snr_threshold/sim_results',
    snrs=snrs,
    snr_thresholds=snr_thresholds
)