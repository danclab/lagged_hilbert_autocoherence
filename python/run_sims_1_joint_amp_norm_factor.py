import numpy as np
import scipy
from scipy.signal import hilbert
from lagged_autocoherence import lagged_hilbert_autocoherence, generate_surrogate, filter_data, lagged_fourier_autocoherence
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
T=10
# Sampling rate
srate=500
time=np.linspace(0,T,T*srate)

# Burst frequency
f1 = 15
# Length of bursts in cycles
f1_num_cycles=5
# Number of bursts
f1_num_bursts=1

n_trials=100

snr_db = -10
lags = np.arange(0.5, 10.5, 0.5)
f_min = 5
f_max = 100

signal = gen_signal_bursts(T, n_trials, srate, f1, f1_num_cycles, f1_num_bursts, snr_db)

freqs, psd = scipy.signal.welch(signal, fs=srate, window='hann',
                                nperseg=srate, noverlap=int(srate / 2), nfft=srate * 2, detrend='constant',
                                return_onesided=True, scaling='density', axis=- 1, average='mean')
idx = (freqs >= f_min) & (freqs <= f_max)
freqs = freqs[idx]
psd = psd[:, idx]

lhc = lagged_hilbert_autocoherence(signal, freqs, lags, srate, n_jobs=-1)

# --- Surrogate-based thresholds ---
filtered_signal = filter_data(signal, srate, freqs[0], freqs[-1], verbose=False)

surr_data = generate_surrogate(filtered_signal, n_shuffles=1000, n_jobs=-1)

joint_energy_surrogates = np.mean(surr_data, axis=-1)
thresholds = np.percentile(joint_energy_surrogates, 95, axis=-1)

df = np.diff(freqs)[0]
n_pts = signal.shape[-1]
T = n_pts * 1 / srate
time = np.linspace(0, T, int(T * srate))
n_lags = len(lags)
padd_signal = np.hstack([np.zeros((n_trials, n_pts)), signal, np.zeros((n_trials, n_pts))])
signal_fft = np.fft.rfft(padd_signal, axis=-1)
fft_frex = np.fft.rfftfreq(padd_signal.shape[-1], d=1 / srate)
sigma = df * .5

def lhc_extended(freq):
    f_num = np.zeros((n_trials, n_lags))
    f_denom = np.zeros((n_trials, n_lags))
    f_lhc = np.zeros((n_trials, n_lags))

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
    analytic_signal = analytic_signal[:, n_pts:2 * n_pts]

    for l_idx, lag in enumerate(lags):
        # Duration of this lag in s
        lag_dur_s = np.max([lag / freq, 1 / srate])

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
        f1 = analytic_signal[:, eval_pts[:-1, np.newaxis] + np.arange(n_range)]
        # Analytic signal at n=1...n_evals evaluation points, and m=0..n_range time points in between
        f2 = analytic_signal[:, eval_pts[1:, np.newaxis] + np.arange(n_range)]

        # Calculate the phase difference and amplitude product
        phase_diff = np.angle(f2) - np.angle(f1)
        amp_prod = np.abs(f1) * np.abs(f2)

        # Lagged autocoherence
        num = np.sum(amp_prod * np.exp(complex(0, 1) * phase_diff), axis=1)
        f1_pow = np.power(f1, 2)
        f2_pow = np.power(f2, 2)
        denom = np.sqrt(np.sum(np.abs(f1_pow), axis=1) * np.sum(np.abs(f2_pow), axis=1))

        lhc = np.abs(num / denom)

        f_num[:, l_idx] = np.mean(np.abs(num), axis=-1)
        f_denom[:, l_idx] = np.mean(denom, axis=-1)
        f_lhc[:, l_idx] = np.mean(lhc, axis=-1)
    return f_num, f_denom, f_lhc


f_num_true, f_denom_true, f_lhc_true = lhc_extended(f1)
f_num_false, f_denom_false, f_lhc_false = lhc_extended(50)

np.savez(
    '../output/sims/joint_amp_norm_factor/sim_results',
    lags=lags,
    freqs=freqs,
    lhc=lhc,
    f_lhc_true=f_lhc_true,
    f_lhc_false=f_lhc_false,
    f_num_true=f_num_true,
    f_num_false=f_num_false,
    f_denom_true=f_denom_true,
    f_denom_false=f_denom_false,
    thresholds=thresholds
)