import glob
import os
import os.path as op

import scipy
from fooof import FOOOF

import utils
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from mne import set_log_level, read_epochs
from mne.io import read_epochs_eeglab
from mne.preprocessing import peak_finder


def process_epoch_alpha_bursts(epoch_path, alpha_range, list_of_sensors=[], name="infant-alpha"):
    subject = epoch_path.parts[-3]
    age_label = processed_data_paths[0].parts[-6]
    if epoch_path.suffix == ".set":
        epochs = read_epochs_eeglab(epoch_path)
    else:
        epochs = read_epochs(epoch_path)

    epoch_id_to_name = {v: k for k, v in epochs.event_id.items()}
    if len(list_of_sensors) > 0:
        ch_names = list_of_sensors
    else:
        ch_names = epochs.info["ch_names"]
    trial_labels = [epoch_id_to_name.get(i) for i in epochs.events[:, 2]]
    trial_nos = np.arange(len(epochs))
    trial_selections = [utils.get_surrounding_elements(trial_nos, i, 9) for i in trial_nos]

    epochs_filtered = epochs.filter(alpha_range[0], alpha_range[1])
    hilbert_phase = epochs_filtered.copy().apply_hilbert(envelope=False)
    hilbert_envelope = epochs_filtered.copy().apply_hilbert(envelope=True)
    hilbert_phase = hilbert_phase.copy().get_data()
    hilbert_phase = np.angle(hilbert_phase)
    hilbert_envelope = hilbert_envelope.get_data()

    data_dict = {
        "burst_start": [],
        "burst_stop": [],
        "burst_duration": [],
        "burst_peak_time": [],
        "burst_peak_abs_amplitude": [],
        "burst_peak_rel_amplitude": [],
        "burst_cycles": [],
        "burst_threshold": [],
        "sensor_name": [],
        "trial_number": [],
        "trial_label": [],
        "subject": [],
        "age_label": []
    }

    chan_trial_list = list(product(np.arange(len(ch_names)), trial_nos))

    for comp_ix, (channel, trial) in enumerate(chan_trial_list):
        trial_selection = trial_selections[trial]
        threshold = np.percentile(hilbert_envelope[trial_selection, channel, :].flatten(), 75, axis=0)
        loc, mag = peak_finder(hilbert_phase[trial][channel], thresh=2.5, verbose=False)
        crossings = np.array(utils.detect_threshold_crossings(hilbert_envelope[trial][channel], threshold))
        for onset, end in crossings:
            peak_ix = np.argmax(hilbert_envelope[trial][channel][onset:end])
            data_dict["burst_start"].append(epochs.times[onset])
            data_dict["burst_stop"].append(epochs.times[end - 1])
            data_dict["burst_duration"].append(epochs.times[end - 1] - epochs.times[onset])
            data_dict["burst_peak_time"].append(epochs.times[onset:end][peak_ix])
            data_dict["burst_peak_abs_amplitude"].append(hilbert_envelope[trial][channel][onset:end][peak_ix])
            data_dict["burst_peak_rel_amplitude"].append(
                hilbert_envelope[trial][channel][onset:end][peak_ix] - threshold)
            data_dict["burst_cycles"].append(utils.count_values_between(loc, onset, end))
            data_dict["burst_threshold"].append(threshold)
            data_dict["sensor_name"].append(ch_names[channel])
            data_dict["trial_number"].append(trial)
            data_dict["trial_label"].append(trial_labels[trial])
            data_dict["subject"].append(subject)
            data_dict["age_label"].append(age_label)

        print(subject, f"{comp_ix + 1}/{len(chan_trial_list)}")

    data_output_dir = output_dir.joinpath(f"{name}_{age_label}_{subject}.csv")
    data_output = pd.DataFrame.from_dict(data_dict)
    data_output.to_csv(data_output_dir, index=False)
    print(subject, "done!")


# Define helper to extract periodic component
def extract_periodic(freqs, psd):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, verbose=False)
    fm.fit(freqs, psd)
    return fm.get_params('peak_params'), fm._peak_fit


def define_fois(freqs, mean_psd, debug=False):
    mean_resid = mean_psd
    foi_pk_freqs = []
    foi_pk_vals = []
    foi_fwhms = []
    foi_ranges = []
    orig_pks, properties = scipy.signal.find_peaks(mean_resid)
    while True:
        pks, properties = scipy.signal.find_peaks(mean_resid)

        pk_vals = mean_resid[pks]
        sorted_idx = np.argsort(-pk_vals)
        pk_vals = pk_vals[sorted_idx]
        pks = pks[sorted_idx]

        pk_idx = pks[0]
        pk_val = pk_vals[0]
        pk_freq = freqs[pk_idx]

        if pk_val < np.std(mean_resid):
            if debug:
                print('Peak val={}, thresh={}'.format(pk_val, .5 * np.std(mean_resid)))
            break

        l_idx = np.where(mean_resid[:pk_idx] <= pk_val * .5)[0]
        r_idx = np.where(mean_resid[pk_idx:] <= pk_val * .5)[0]

        if len(l_idx) and len(r_idx):
            l_freq = freqs[l_idx[-1]]
            r_freq = freqs[pk_idx + r_idx[0]]
            r_side = (r_freq - pk_freq)
            l_side = (pk_freq - l_freq)
            fwhm = 2 * np.min([r_side, l_side])
        elif len(l_idx):
            l_freq = freqs[l_idx[-1]]
            fwhm = 2 * (pk_freq - l_freq)
        elif len(r_idx):
            r_freq = freqs[pk_idx + r_idx[0] + 1]
            fwhm = 2 * (r_freq - pk_freq)

        l_freq = pk_freq - fwhm * .5
        r_freq = pk_freq + fwhm * .5
        sd = fwhm / (2 * np.sqrt(2 * np.log(2)))
        A = pk_vals[0] * np.exp(-.5 * ((freqs - pk_freq) / sd) ** 2)
        nearest_orig = np.min(np.abs(freqs[np.array(orig_pks)] - pk_freq))

        if pk_val > 1.5 * np.std(mean_resid) and (
                (pk_freq < 10 and fwhm > 1) or (pk_freq >= 10 and fwhm > 3)) and nearest_orig < 3:
            print('Peak: freq={}, val={}, width={}, range={}-{}'.format(pk_freq, pk_val, fwhm, pk_freq - fwhm * .5,
                                                                        pk_freq + fwhm * .5))
            print('Distance to nearest original peak={}'.format(nearest_orig))

            foi_pk_freqs.append(pk_freq)
            foi_pk_vals.append(pk_val)
            foi_fwhms.append(fwhm)
            foi_ranges.append([l_freq, r_freq])
        elif debug:
            if not ((pk_freq < 10 and fwhm > 1) or (pk_freq >= 10 and fwhm > 3)):
                print('Peak: freq={}, fhwm={} to narrow'.format(pk_freq, fwhm))
            if pk_val <= 1.5 * np.std(mean_resid):
                print('Peak: freq={} too low amplitude'.format(pk_freq))
            if nearest_orig >= 3:
                print('Peak: freq={} too far from original peaks'.format(pk_freq))

        mean_resid = mean_resid - A
        # mean_resid[mean_resid<0]=0
    sorted_idx = np.argsort(foi_pk_freqs)

    foi_pk_freqs = np.array(foi_pk_freqs)
    foi_pk_vals = np.array(foi_pk_vals)
    foi_fwhms = np.array(foi_fwhms)
    foi_ranges = np.array(foi_ranges)

    foi_pk_freqs = foi_pk_freqs[sorted_idx]
    foi_pk_vals = foi_pk_vals[sorted_idx]
    foi_fwhms = foi_fwhms[sorted_idx]
    foi_ranges = foi_ranges[sorted_idx, :]

    return foi_pk_freqs, foi_pk_vals, foi_fwhms, foi_ranges


def determine_alpha_range(age):
    epochs = {
        'exe': ['EBM', 'LEXT', 'FTGE', 'EXGC', 'EXEND'],
        'obs': ['OBM', 'LOBS', 'FTGO', 'OBGC', 'OBEND']
    }
    freq_lims = [1, 50]

    out_path = op.join('/home/bonaiuto/lagged_hilbert_coherence/output/dev_beta_umd', age)

    all_psd = []
    sub_paths = glob.glob(op.join(out_path, 'sub-*'))
    subjects_included=[]
    for sub_path in sub_paths:
        subj_id = os.path.split(sub_path)[-1]

        epo_psd = []
        has_all = True
        for condition in ['exe', 'obs']:
            for epo in epochs[condition]:
                data_fname = op.join(sub_path, f'{subj_id}-{condition}-{epo}.npz')
                if not op.exists(data_fname):
                    has_all = False
                    break
                else:
                    res = dict(np.load(data_fname))
                    freqs = res['freqs']
                    # Average PSD over channels
                    mean_psd = np.mean(res['psd'], axis=0)
                    # Average LC over trials then channels
                    f_idx = (freqs >= freq_lims[0]) & (freqs <= freq_lims[1])
                    freqs = freqs[f_idx]
                    mean_psd = mean_psd[f_idx]
                    epo_psd.append(mean_psd)
            if not has_all:
                break
        if has_all:
            print(f'{age}: {subj_id}')
            all_psd.append(epo_psd)
            subjects_included.append(subj_id)
    all_psd = np.array(all_psd)

    m_psd=np.mean(all_psd,axis=1)
    periodic=[]
    for s_idx in range(m_psd.shape[0]):
        _, p = extract_periodic(freqs, m_psd[s_idx,:])
        periodic.append(p)
    periodic=np.array(periodic)
    m_periodic=np.mean(periodic,axis=0)

    foi_pk_freqs, foi_pk_vals, foi_fwhms, foi_ranges = define_fois(freqs, m_periodic)
    foi_ranges = foi_ranges[np.argsort(foi_ranges[:, 0])]
    alpha_range = None
    for i in range(foi_ranges.shape[0]):
        if foi_ranges[i, 0] > 3:
            alpha_range = foi_ranges[i, :]
            break
    print(f'{age} alpha: {alpha_range}')
    return subjects_included, alpha_range

if __name__ == '__main__':

    set_log_level(verbose="ERROR")

    output_dir = Path("../output/dev_beta_umd/alpha_bursts/")

    ages=['9m','12m','adult']
    for age in ages:
        dataset_dir = Path("/home/common/bonaiuto/dev_beta_umd/data/"+age+"/derivatives/NEARICA_behav/")
        processed_data_paths = utils.get_files(
            dataset_dir, "*.set",
            strings=["tool", "obs", "exe", "eeg", "processed_data"]
        )
        subjects_included, alpha_range=determine_alpha_range(age)
        for epoch_path in processed_data_paths:
            subject = epoch_path.parts[-3]
            if subject in subjects_included:
                process_epoch_alpha_bursts(epoch_path, alpha_range)

