import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.decomposition import PCA
import os
import json

# =============================================================================
# Preprocessing steps — each is a factory that returns a (Raw -> Raw) callable
# =============================================================================

def notch(freq: float = 60.0):
    """Remove power line noise."""
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        raw.notch_filter(freq)
        return raw
    apply._desc = f"notch(freq={freq})"
    return apply


def bandpass(bands: list[tuple]):
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        raw._band_data = {}
        for low_cut, high_cut in bands:
            filtered = raw.copy()
            filtered.filter(l_freq=low_cut, h_freq=high_cut)
            raw._band_data[(low_cut, high_cut)] = filtered
        return raw
    apply._desc = f"bandpass(bands={bands})"
    return apply


def rereference(ref: str = "average"):
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        raw.set_eeg_reference(ref_channels=ref, projection=False)
        return raw
    apply._desc = f"rereference(ref={ref!r})"
    return apply


def epoch(tmin: float = 0.0, tmax: float = 3.0,
          baseline: tuple = None, event_id: dict = None):
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        events_array, event_id_dict = mne.events_from_annotations(raw)
        selected_ids = event_id if event_id is not None else event_id_dict
        raw._epochs = mne.Epochs(
            raw, events_array,
            event_id=selected_ids,
            tmin=tmin, tmax=tmax,
            baseline=baseline,
            preload=True
        )
        raw._event_id = selected_ids  # store for remapping later
        print(f"Epoched: {len(raw._epochs)} trials  |  classes: {list(selected_ids.keys())}")
        return raw
    apply._desc = f"epoch(tmin={tmin}, tmax={tmax})"
    return apply


def add_idle_class(window_dur: float = 3.0, idle_start_min: float = 1.0, label: str = 'idle'):
    """Add idle-class annotations from unannotated time after idle_start_min minutes.

    Must be added to the pipeline BEFORE epoch().
    """
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        idle_start_sec = idle_start_min * 60.0
        recording_end_sec = raw.times[-1]

        # build busy intervals from existing annotations, padded by window_dur
        busy_intervals = [
            (ann['onset'], ann['onset'] + ann['duration'] + window_dur)
            for ann in raw.annotations
        ]

        # walk through recording and collect non-overlapping idle windows
        idle_onsets = []
        t = idle_start_sec
        while t + window_dur <= recording_end_sec:
            window_end = t + window_dur
            overlaps = any(
                not (window_end <= busy_start or t >= busy_end)
                for busy_start, busy_end in busy_intervals
            )
            if not overlaps:
                idle_onsets.append(t)
                t += window_dur
            else:
                t += 0.1

        if not idle_onsets:
            print(f"[idle] Warning: no idle windows found after {idle_start_min} min")
            return raw

        idle_annotations = mne.Annotations(
            onset=np.array(idle_onsets),
            duration=np.full(len(idle_onsets), window_dur),
            description=np.array([label] * len(idle_onsets)),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + idle_annotations)
        print(f"[idle] Added {len(idle_onsets)} idle windows  |  label: '{label}'")
        return raw

    apply._desc = f"add_idle_class(window_dur={window_dur}, idle_start_min={idle_start_min})"
    return apply


def plot_psd(title: str = "PSD", fmin: float = 0.1, fmax: float = 150.0):
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch]
        psd, freqs = mne.time_frequency.psd_array_welch(
            raw.get_data(), fmin=fmin, fmax=fmax,
            n_fft=2048, sfreq=raw.info['sfreq']
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        for i, ch in enumerate(eeg_channels):
            ax.semilogy(freqs, psd[i], label=ch, alpha=0.6)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (V²/Hz)')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return raw
    apply._desc = f"plot_psd(title={title!r})"
    return apply

def add_fake_jaw_clench(onset: float = 10.0, duration: float = 3.0):
    def apply(raw):
        fake = mne.Annotations(
            onset=[onset],
            duration=[duration],
            description=['jaw_clench'],
            orig_time=raw.annotations.orig_time  # match existing orig_time
        )
        raw.set_annotations(raw.annotations + fake)
        return raw
    apply._desc = f"add_fake_jaw_clench(onset={onset}, duration={duration})"
    return apply

# =============================================================================
# Feature extractors — take a processed Raw, return (X, y)
# =============================================================================

def _get_eeg_channels(raw):
    exclude = {'Trigger', 'Event'}
    return [ch for ch in raw.ch_names if ch not in exclude]


def extract_csp(n_components: int = 6):
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        epochs = raw._epochs
        y = epochs.events[:, 2]

        if hasattr(raw, '_band_data') and raw._band_data:
            events_array, event_id_dict = mne.events_from_annotations(raw)
            band_features = []
            for (low, high), filtered_raw in raw._band_data.items():
                band_epochs = mne.Epochs(
                    filtered_raw, events_array,
                    event_id=event_id_dict,
                    tmin=epochs.tmin, tmax=epochs.tmax,
                    baseline=None, preload=True
                )
                X_band = band_epochs.get_data(picks=eeg_channels)
                csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
                band_features.append(csp.fit_transform(X_band, y))
            X = np.concatenate(band_features, axis=1)
        else:
            X_raw = epochs.get_data(picks=eeg_channels)
            csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
            X = csp.fit_transform(X_raw, y)

        print(f"[CSP] shape: {X.shape}")
        return X, y
    extract._desc = f"extract_csp(n_components={n_components})"
    return extract


def epoch_filter(bands: list[tuple], notch_freq: float = None, ref: str = None):
    """Apply filtering/rereference per epoch (use INSTEAD of whole-file steps).

    Add this step AFTER epoch() in the pipeline when per_epoch=True.
    bands   — list of (l_freq, h_freq) tuples, same format as bandpass()
    notch_freq — power-line frequency to notch, or None to skip
    ref        — rereferencing scheme ('average', etc.), or None to skip
    """
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        assert hasattr(raw, '_epochs'), "Add epoch() before epoch_filter()."
        if notch_freq is not None:
            data = raw._epochs.get_data()
            data = mne.filter.notch_filter(data, Fs=raw._epochs.info['sfreq'], freqs=notch_freq, verbose=False)
            raw._epochs._data = data
        for l_freq, h_freq in bands:
            raw._epochs.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        if ref is not None:
            raw._epochs.set_eeg_reference(ref_channels=ref, projection=False, verbose=False)
        return raw
    apply._desc = f"epoch_filter(bands={bands}, notch_freq={notch_freq}, ref={ref!r})"
    return apply


def extract_tensor(scale: bool = True, per_epoch_scale: bool = True):
    """Extract epoch data as a float32 tensor.

    scale           — whether to z-score at all
    per_epoch_scale — True (default): z-score each trial independently (per channel)
                      False: z-score using global mean/std across all trials
    """
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        X = raw._epochs.get_data(picks=eeg_channels)
        y = raw._epochs.events[:, 2]
        if scale:
            if per_epoch_scale:
                mean = X.mean(axis=-1, keepdims=True)          # (n_trials, n_ch, 1)
                std  = X.std(axis=-1, keepdims=True) + 1e-8
            else:
                mean = X.mean(axis=(0, -1), keepdims=True)     # (1, n_ch, 1) — global
                std  = X.std(axis=(0, -1), keepdims=True) + 1e-8
            X = (X - mean) / std
        X = X[:, np.newaxis, :, :].astype(np.float32)
        print(f"[Tensor] shape: {X.shape}  scale={'per-epoch' if per_epoch_scale else 'global'}")
        return X, y
    extract._desc = f"extract_tensor(scale={scale}, per_epoch_scale={per_epoch_scale})"
    return extract


def extract_cwt(freqs: np.ndarray = None, n_cycles: float = 6.0):
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        _freqs = freqs if freqs is not None else np.arange(1, 101, 1.0)
        raw_X = raw._epochs.get_data(picks=eeg_channels)
        y = raw._epochs.events[:, 2]
        sfreq = raw.info['sfreq']
        X_cwt = mne.time_frequency.tfr_array_morlet(
            raw_X, sfreq=sfreq, freqs=_freqs,
            n_cycles=n_cycles, output='power'
        ).astype(np.float32)
        print(f"[CWT] shape: {X_cwt.shape}")
        return X_cwt, y
    extract._desc = f"extract_cwt(n_cycles={n_cycles})"
    return extract


def extract_bandpower(bands: dict = None):
    _bands = bands or {
        'theta': (4,  8),
        'alpha': (8,  13),
        'beta':  (13, 30),
        'gamma': (30, 55),
        'emg':   (65, 100),
    }
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        raw_X = raw._epochs.get_data(picks=eeg_channels)
        y = raw._epochs.events[:, 2]
        sfreq = raw.info['sfreq']
        n_trials, n_ch, n_times = raw_X.shape
        features = []
        for t in range(n_trials):
            trial_feats = []
            for c in range(n_ch):
                psd, freqs = mne.time_frequency.psd_array_welch(
                    raw_X[t, c][np.newaxis, :], sfreq=sfreq,
                    fmin=1, fmax=150, n_fft=min(n_times, 256)
                )
                for lo, hi in _bands.values():
                    idx = (freqs >= lo) & (freqs <= hi)
                    trial_feats.append(np.log(psd[0, idx].mean() + 1e-10))
            features.append(trial_feats)
        X = np.array(features, dtype=np.float32)
        print(f"[Bandpower] shape: {X.shape}  bands: {list(_bands.keys())}")
        return X, y
    extract._desc = f"extract_bandpower(bands={list(_bands.keys())})"
    return extract

def extract_pca(n_components: int = 10):
    """PCA dimensionality reduction on epoch tensor.

    Output shape: (n_trials, n_components) — float32.
    """
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        X = raw._epochs.get_data(picks=eeg_channels)  # (n_trials, n_ch, n_times)
        y = raw._epochs.events[:, 2]

        # flatten each trial to a 1D vector before PCA
        n_trials = X.shape[0]
        X_flat = X.reshape(n_trials, -1)  # (n_trials, n_ch * n_times)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_flat).astype(np.float32)

        explained = pca.explained_variance_ratio_.sum()
        print(f"[PCA] shape: {X_pca.shape}  |  variance explained: {explained:.2%}")
        return X_pca, y

    extract._desc = f"extract_pca(n_components={n_components})"
    return extract

def extract_raw():
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        X = raw._epochs.get_data(picks=eeg_channels).astype(np.float32)
        y = raw._epochs.events[:, 2]
        print(f"[Raw] shape: {X.shape}")
        return X, y
    extract._desc = "extract_raw()"
    return extract

# =============================================================================
# Pipeline
# =============================================================================

class MNEPipeline:
    def __init__(self, fif_file: str):
        self.fif_file = fif_file
        self.steps = []

    def add(self, step) -> 'MNEPipeline':
        self.steps.append(step)
        return self

    def describe(self):
        print(f"Pipeline: {self.fif_file}")
        for i, step in enumerate(self.steps):
            print(f"  {i+1}. {repr(step)}")

    def run(self, extractor=None):
        raw = mne.io.read_raw_fif(self.fif_file, preload=True)
        for step in self.steps:
            print(f"Applying: {repr(step)}")
            raw = step(raw)
        if extractor is not None:
            return extractor(raw)
        return raw

    def to_dict(self):
        return {'steps': [getattr(s, '_desc', repr(s)) for s in self.steps]}


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # parameters and file paths here
    files = [
        './data_collection/annotated_eeg/chengyi0210.fif',
        './data_collection/annotated_eeg/pilapil0226.fif',
    ]
    save_dir = './data_collection/preprocessed_data'
    preprocessed_data_filename = 'chengyi.npz'
    label_map = {'idle': 0, 'move': 1, 'jaw_clench': 2}
    
    # define the extractor once so its repr can be captured in the config
    extractor = extract_raw()

    all_X, all_y = [], []

    for file in files:
        pipeline = MNEPipeline(file)
        pipeline.add(notch(60))
        pipeline.add(bandpass([(8, 13), (13, 30)]))
        pipeline.add(rereference())
        # always add this add idle class and the epoch class
        pipeline.add(add_idle_class(window_dur=3.0, idle_start_min=1.0))
        pipeline.add(add_fake_jaw_clench())
        pipeline.add(epoch(tmin=0, tmax=3))
        pipeline.describe()

        # run without extractor to keep the raw object (for event_id access)
        raw_processed = pipeline.run()
        X, y = extractor(raw_processed)

        # remap MNE's auto-assigned ids to consistent 0/1/2 labels
        auto_ids = raw_processed._event_id  # e.g. {'idle': 1, 'jaw_clench': 2, 'move': 3}
        remap = {v: label_map[k] for k, v in auto_ids.items() if k in label_map}
        y = np.array([remap[yi] for yi in y])

        print(X.shape, np.unique(y))
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Total: {X.shape}, labels: {np.unique(y)}")

    save_path = os.path.join(save_dir, preprocessed_data_filename) if save_dir else 'dataset.npz'
    np.savez(save_path, X=X, y=y)
    print(f"Saved {X.shape[0]} trials to {save_path}")

    config = {
        'files': files,
        'label_map': label_map,
        'X_shape': list(X.shape),
        'y_shape': list(y.shape),
        'pipeline': pipeline.to_dict()['steps'],
        'extractor': getattr(extractor, '_desc', repr(extractor)),
    }
    config_path = save_path.replace('.npz', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")