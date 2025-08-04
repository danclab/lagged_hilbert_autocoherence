import os
import os.path as op
import glob

import scipy
from mne.io import read_epochs_eeglab
from lagged_autocoherence import *

base_path = '/home/common/bonaiuto/dev_beta_umd/data'
out_dir='/home/bonaiuto/lagged_hilbert_coherence/output/dev_beta_umd/'

ages=['9m','12m','adult']
pipeline='NEARICA_behav'
c_cluster_chans=['E16', 'E20', 'E21', 'E22','E41', 'E49', 'E50', 'E51']
n_chans=len(c_cluster_chans)
condition_epochs={
    'exe': ['EBM','LEXT','FTGE','EXGC','EXEND'],
    'obs': ['OBM','LOBS','FTGO','OBGC','OBEND']
}

freq_lims = [4, 100]
lags=np.arange(0.1,4.5,.1)

for age in ages:
    sub_paths = sorted(glob.glob(op.join(base_path, age, 'derivatives', 'NEARICA_behav', 'sub-*')))
    for subject_path in sub_paths:
        subj_id = os.path.split(subject_path)[-1]
        subject_data_fname = op.join(subject_path, 'processed_data',
                                     f'{subj_id}_task-tool_obs_exe_eeg_processed_data.set')
        if op.exists(subject_data_fname):
            EEG = read_epochs_eeglab(subject_data_fname)

            fs = EEG.info['sfreq']

            data = EEG.get_data()
            n_epochs = data.shape[0]
            n_chans = data.shape[1]
            n_samps = data.shape[2]

            has_all=True
            for condition in condition_epochs:
                epochs = condition_epochs[condition]
                for epo in epochs:
                    if epo in EEG.event_id:
                        event_id = EEG.event_id[epo]
                        trial_idx = np.where(EEG.events[:, 2] == event_id)[0]
                        if len(trial_idx) < 5:
                            has_all=False
                    else:
                        has_all=False

            if has_all:
                out_path = os.path.join(out_dir, age, subj_id)
                os.makedirs(out_path, exist_ok=True)

                for condition in condition_epochs:
                    epochs = condition_epochs[condition]
                    for epo in epochs:
                        if epo in EEG.event_id:
                            event_id = EEG.event_id[epo]
                            trial_idx = np.where(EEG.events[:, 2] == event_id)[0]
                            if len(trial_idx) >= 5:
                                psd = []
                                lfc = []
                                lhc = []
                                for c_idx, c in enumerate(c_cluster_chans):
                                    c_chan_idx = EEG.info['ch_names'].index(c)

                                    freqs, ch_psd = scipy.signal.welch(data[trial_idx, c_chan_idx, :], fs=fs, window='hann',
                                                                       nperseg=int(fs), noverlap=int(fs / 2),
                                                                       nfft=int(fs * 2),
                                                                       detrend='constant',
                                                                       return_onesided=True, scaling='density', axis=- 1,
                                                                       average='mean')
                                    f_idx = (freqs >= freq_lims[0]) & (freqs <= freq_lims[-1])
                                    freqs = freqs[f_idx]
                                    ch_psd = np.mean(ch_psd[:, f_idx], axis=0)

                                    ch_lfc = lagged_fourier_autocoherence(data[trial_idx, c_chan_idx, :], freqs, lags, fs,
                                                                          n_jobs=-1)
                                    ch_lhc = lagged_hilbert_autocoherence(data[trial_idx, c_chan_idx, :], freqs, lags, fs,
                                                                          surr_method='phase', n_jobs=-1)

                                    psd.append(ch_psd)
                                    lfc.append(ch_lfc)
                                    lhc.append(ch_lhc)
                                psd = np.array(psd)
                                lfc = np.array(lfc)
                                lhc = np.array(lhc)

                                out_fname = os.path.join(out_path, '{}-{}-{}.npz'.format(subj_id, condition, epo))
                                np.savez(out_fname,
                                         **{
                                             'freqs': freqs,
                                             'lags': lags,
                                             'psd': psd,
                                             'lfc': lfc,
                                             'lhc': lhc
                                         })