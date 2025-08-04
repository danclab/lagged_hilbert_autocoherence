import numpy as np

def many_is_in(multiple, target):
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return any(check_)


def compute_rms(time_series):
    return np.sqrt(np.mean(np.square(time_series)))


def scale_noise(signal_without_noise, noise, desired_snr_db):
    # Compute power of the original signal
    Ps = compute_rms(signal_without_noise) ** 2

    # Convert desired SNR from dB to linear scale
    snr_linear = 10 ** (desired_snr_db / 10)

    # Calculate the desired noise power based on the desired SNR
    desired_noise_power = Ps / snr_linear

    # Compute scaling factor for the noise
    alpha = np.sqrt(desired_noise_power / compute_rms(noise) ** 2)

    return alpha * noise


def sigmoid(x, x0, k):
    L=-1
    b=1
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)