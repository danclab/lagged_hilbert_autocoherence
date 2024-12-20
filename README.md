# Lagged Hilbert autocoherence
Lagged Hilbert autocoherence, phase-locking value, and amplitude autocoherence. Library accompanying 
the paper Zhang et al. (2024) "Multi-scale parameterization of neural rhythmicity with lagged Hilbert 
autocoherence"

<https://www.biorxiv.org/content/10.1101/2024.12.05.627017v2.abstract>


## Requirements
python version: joblib, scipy, numpy, statsmodels (>=0.13.5)

matlab version: parallel processing toolbox, signal processing toolbox

# Python files
- lagged_autocoherence.py: core functions
- demo.ipynb: demonstration of the new lagged autocoherence algorithm
- multi_trial_demo.ipynb: demonstration of new lagged autocoherence algorithm with multiple trials
- sims.ipynb: tests with simulated data
- meg_sensor_data.ipynb: analysis of MEG sensor data
- lfp_data.ipynb: analysis of LFP monkey data

# Matlab files
- lagged_hilbert_autocoherence.m: core function
- ar_surr.m: ARMA surrogate data generation
- demo.m: demonstration of new lagged autocoherence algorithm
- multi_trial_demo.m: demonstration of new lagged autocoherence algorithm with multiple trials
- rfft.m: Fourier transform of real signal
- irfft.m: inverse Fourier transform of real signal
- hilbert.m: hilbert transform
