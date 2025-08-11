from pathlib import Path

import numpy as np


def check_many(multiple, target, func=None):
    """
    Checks for a presence of strings in a target strings.

    Parameters:
    multiple (list): strings to be found in target string
    target (str): target string
    func (str): "all" or "any", use the fuction to search for any or all strings in the filename.

    Notes:
    - this function works really well with if statement for list comprehension
    """
    func_dict = {
        "all": all, "any": any
    }
    if func in func_dict.keys():
        use_func = func_dict[func]
    elif func == None:
        raise ValueError("pick function 'all' or 'any'")
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return use_func(check_)


def get_files(target_path, suffix, strings=[""], prefix=None, check="all", depth="all"):
    """
    Returns a list of the files with specific extension, prefix and name containing
    specific strings. Either all files in the directory or in this directory.

    Parameters:
    target path (str or pathlib.Path or os.Path): the most shallow searched directory
    suffix (str): file extension in "*.ext" format
    strings (list of str): list of strings searched in the file name
    prefix (str): limit the output list to the file manes starting with prefix
    check (str): "all" or "any", use the fuction to search for any or all strings in the filename.
    depth (str): "all" or "one", depth of search (recurrent or shallow)

    Notes:
    - returns a list of pathlib.Path objects
    """
    path = Path(target_path)
    if depth == "all":
        subdirs = [subdir for subdir in path.rglob(suffix) if check_many(strings, str(subdir.name), check)]
        subdirs.sort()
        if isinstance(prefix, str):
            subdirs = [subdir for subdir in subdirs if subdir.name.startswith(prefix)]
        return subdirs
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir() if
                   all([subdir.is_file(), subdir.suffix == suffix[1:], check_many(strings, str(subdir.name), check)])]
        if isinstance(prefix, str):
            subdirs = [subdir for subdir in subdirs if subdir.name.startswith(prefix)]
        subdirs.sort()
        return subdirs


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


def get_surrounding_elements(arr, target_index, n):
    """
    Returns `n` elements surrounding the `target_index` in the array.
    Includes the element at `target_index`, centered if possible.
    If near boundaries, shifts the window to keep `n` elements.
    """
    arr_len = len(arr)

    if n > arr_len:
        raise ValueError("n cannot be larger than the length of the array")

    half = n // 2

    # Compute start and end with boundaries in mind
    start = max(0, target_index - half)
    end = start + n

    # Adjust if end goes out of bounds
    if end > arr_len:
        end = arr_len
        start = end - n

    return arr[start:end]


def detect_threshold_crossings(signal, threshold, direction='above'):
    """
    Detect onset and end indices of parts of the signal that cross a threshold.

    Parameters:
    - signal (np.ndarray): 1D array of signal values
    - threshold (float): The threshold to detect crossings
    - direction (str): 'above' to detect when signal > threshold,
                       'below' to detect when signal < threshold

    Returns:
    - crossings (list of tuples): List of (onset_index, end_index) tuples
    """
    if direction == 'above':
        condition = signal > threshold
    elif direction == 'below':
        condition = signal < threshold
    else:
        raise ValueError("direction must be 'above' or 'below'")

    condition = np.asarray(condition, dtype=int)
    diff = np.diff(condition)

    # Onset: 0 -> 1 transition; End: 1 -> 0 transition
    onsets = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases: if signal starts above threshold
    if condition[0]:
        onsets = np.insert(onsets, 0, 0)
    # if signal ends above threshold
    if condition[-1]:
        ends = np.append(ends, len(signal))

    crossings = list(zip(onsets, ends))
    return crossings


def count_values_between(values, lower, upper, inclusive=True):
    """
    Count how many values are between lower and upper bounds.

    Parameters:
    - values (list or np.ndarray): List or array of numbers
    - lower (float or int): Lower bound
    - upper (float or int): Upper bound
    - inclusive (bool): If True, include bounds (default: True)

    Returns:
    - int: Number of values within the bounds
    """
    values = np.array(values)

    if inclusive:
        condition = (values >= lower) & (values <= upper)
    else:
        condition = (values > lower) & (values < upper)

    return np.sum(condition)