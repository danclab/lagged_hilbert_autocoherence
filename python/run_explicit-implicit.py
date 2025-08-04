import glob
import os
import scipy
from lagged_autocoherence import *
import mne

from python.utils import many_is_in

freq_lims = [4, 100]
lags=np.arange(0.1,4.5,.1)

sens = ["MLC1", "MLC25", "MLC32", "MLC42", "MLC54", "MLC55", "MLC63"]

base_dir='/home/common/mszul/explicit_implicit_beta/derivatives/processed/'
out_dir='/home/bonaiuto/lagged_hilbert_coherence/output/explicit-implicit/'

sub_paths = sorted(glob.glob(os.path.join(base_dir, 'sub-*')))
for sub_idx in range(len(sub_paths)):
    sub_path = sub_paths[sub_idx]
    sub = os.path.split(sub_path)[-1]
    sub_out_path = os.path.join(out_dir, sub)
    if not os.path.exists(sub_out_path):
        os.mkdir(sub_out_path)

    epo_paths = sorted(glob.glob(os.path.join(sub_path, '{}-*-epo.fif'.format(sub))))
    for epo_path in epo_paths:
        block_num = os.path.split(epo_path)[-1].split('-')[2]
        epo_type = os.path.split(epo_path)[-1].split('-')[3]
        if epo_type == 'visual' or epo_type == 'motor':
            epo = mne.read_epochs(epo_path)
            epo = epo.pick_types(meg=True, ref_meg=False, misc=False)

            channels_used = [i for i in epo.info.ch_names if many_is_in(["MLC"], i)]
            channels_used = [i for i in channels_used if not many_is_in(sens, i)]

            fs = epo.info['sfreq']
            epo_data = epo.get_data()
            epo_data = np.moveaxis(epo_data, [0, 1, 2], [1, 0, 2])

            psd = []
            lfc = []
            lhc = []
            chs = []
            for ch_idx in range(epo_data.shape[0]):
                if epo.info['ch_names'][ch_idx] in channels_used:
                    data = epo_data[ch_idx, :, :]
                    chs.append(ch_idx)

                    freqs, ch_psd = scipy.signal.welch(data, fs=fs, window='hann',
                                                       nperseg=int(fs), noverlap=int(fs / 2), nfft=int(fs * 2),
                                                       detrend='constant',
                                                       return_onesided=True, scaling='density', axis=- 1,
                                                       average='mean')
                    f_idx = (freqs >= freq_lims[0]) & (freqs <= freq_lims[-1])
                    freqs = freqs[f_idx]
                    ch_psd = np.mean(ch_psd[:, f_idx], axis=0)

                    ch_lfc = lagged_fourier_autocoherence(data, freqs, lags, fs, n_jobs=-1)
                    ch_lhc = lagged_hilbert_autocoherence(data, freqs, lags, fs, surr_method='phase', n_jobs=-1)

                    psd.append(ch_psd)
                    lfc.append(ch_lfc)
                    lhc.append(ch_lhc)
            chs = np.array(chs)
            psd = np.array(psd)
            lfc = np.array(lfc)
            lhc = np.array(lhc)

            out_fname = os.path.join(sub_out_path, '{}-{}-{}.npz'.format(sub, block_num, epo_type))
            np.savez(out_fname,
                     **{
                         'chs': chs,
                         'freqs': freqs,
                         'lags': lags,
                         'psd': psd,
                         'lfc': lfc,
                         'lhc': lhc
                     })