# Lagged Hilbert autocoherence (LHaC)
Lagged Hilbert autocoherence, phase-locking value, and amplitude autocoherence. Library accompanying
the paper Zhang et al. (2024) "Multi-scale parameterization of neural rhythmicity with lagged Hilbert
autocoherence"

<https://www.biorxiv.org/content/10.1101/2024.12.05.627017v2.abstract>


## Requirements
python version: joblib, scipy, numpy, MNE

matlab version: parallel processing toolbox, signal processing toolbox

# Python files
- lagged_autocoherence.py: Core functions
- demo.ipynb: Demonstration of the new lagged autocoherence algorithm
- multi_trial_demo.ipynb: Demonstration of new lagged autocoherence algorithm with multiple trials

# Matlab files
- lagged_hilbert_autocoherence.m: Core function
- generate_surrogate.m: Phase-shuffled surrogate data generation
- demo.m: Demonstration of new lagged autocoherence algorithm
- multi_trial_demo.m: Demonstration of new lagged autocoherence algorithm with multiple trials
- rfft.m: Fourier transform of real signal
- irfft.m: Inverse Fourier transform of real signal
- hilbert.m: Hilbert transform

# Analyses from Zhang et al. (2024)
- python/run_sims_0_surrogate_comparison.py: Compare phase-shuffled and ARMA surrogates
- python/run_sims_1_joint_amp_norm_factor.py: Investigate joint amplitude normalization factor inflation
- python/run_sims_2_oscillation.py: Run oscillation simulations
- python/run_sims_3_burst_duration.py: Run simulations varying burst duration
- python/run_sims_4_burst_number.py: Run simulations varying burst number
- R/analyze_sims.R: Analyze simulations at -5 dB SNR
- R/analyze_sims_snr.R: Analyze simulations across SNR levels
- python/run_dev_umd.py: Run LHaC on infant and adult EEG data
- python/run_dev_umd_burst_detection.py: Run amplitude thresholding-based burst detection on infant and adult EEG data
- R/analyze_dev_umd_alpha_bursts.R: Analyze duration of alpha bursts in infant and adult EEG data
- python/run_explicit-implicit.py: Run LHaC on adult MEG data
- R/analyze_explicit-implicit.R: Analyze alpha and beta crossover point and decay rate in adult MEG data