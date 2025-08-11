import sys
import utils
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from mne import set_log_level, read_epochs
from mne.io import read_epochs_eeglab
from mne.preprocessing import peak_finder


def process_epoch_alpha_bursts(epoch_path, list_of_sensors=[], name="infant-alpha"):
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

    epochs_filtered = epochs.filter(7, 13)
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


if __name__ == '__main__':

    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no file index")

    set_log_level(verbose="ERROR")

    dataset_dir = Path("/home/common/bonaiuto/dev_beta_umd/data/adult/derivatives/NEARICA_behav/")
    output_dir = Path("/home/common/mszul/infant_alpha_bursts")
    processed_data_paths = utils.get_files(
        dataset_dir, "*.set",
        strings=["tool", "obs", "exe", "eeg", "processed_data"]
    )

    epoch_path = processed_data_paths[index]

    process_epoch_alpha_bursts(epoch_path)

