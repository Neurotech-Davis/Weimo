import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter


# =============================================================================
# Preprocessing steps — each is a factory that returns a (Raw -> Raw) callable
# =============================================================================

def notch(freq: float = 60.0):
    """Remove power line noise."""
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        raw.notch_filter(freq)
        return raw
    apply.__repr__ = lambda: f"notch(freq={freq})"
    return apply


def bandpass(bands: list[tuple]):
    """Bandpass filter into one or more frequency bands.

    Stores filtered copies keyed by (low, high) on the raw object so that
    downstream feature extractors can access each band separately.

    Parameters
    ----------
    bands : list of (low_cut, high_cut) tuples
        e.g. [(8, 13), (13, 30)] for alpha + beta
        e.g. [(4, 100)] for a single wide pass
    """
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        raw._band_data = {}
        for low_cut, high_cut in bands:
            filtered = raw.copy()
            filtered.filter(l_freq=low_cut, h_freq=high_cut)
            raw._band_data[(low_cut, high_cut)] = filtered
        return raw
    apply.__repr__ = lambda: f"bandpass(bands={bands})"
    return apply


def rereference(ref: str = "average"):
    """Re-reference to common average or a named channel."""
    def apply(raw: mne.io.Raw) -> mne.io.Raw:
        raw.set_eeg_reference(ref_channels=ref, projection=False)
        return raw
    apply.__repr__ = lambda: f"rereference(ref={ref!r})"
    return apply


def epoch(tmin: float = 0.0, tmax: float = 3.0,
          baseline: tuple = None, event_id: dict = None):
    """Epoch the continuous data around annotation onsets.

    Stores the resulting Epochs object as raw._epochs so feature extractors
    can access it.
    """
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
        print(f"Epoched: {len(raw._epochs)} trials  |  classes: {list(selected_ids.keys())}")
        return raw
    apply.__repr__ = lambda: f"epoch(tmin={tmin}, tmax={tmax})"
    return apply


def plot_psd(title: str = "PSD", fmin: float = 0.1, fmax: float = 150.0):
    """Passthrough step that plots the PSD at this point in the pipeline."""
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
    apply.__repr__ = lambda: f"plot_psd(title={title!r})"
    return apply


# =============================================================================
# Feature extractors — take a processed Raw, return (X, y)
# =============================================================================

def _get_eeg_channels(raw):
    return [ch for ch in raw.ch_names if "EEG" in ch]


def extract_csp(n_components: int = 6):
    """CSP log-variance features → CSP + LDA.

    If bandpass() was called with multiple bands, fits a separate CSP per band
    and concatenates — giving (n_trials, n_bands * n_components).
    """
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
    return extract


def extract_tensor(scale: bool = True):
    """Raw epoch tensor → all DL models (EEGNet / ATCNet / CTNet / Conformer).

    Output shape: (n_trials, 1, n_channels, n_times) — float32.
    The leading 1 is the image-channel dimension expected by 2D convolutions.
    """
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        X = raw._epochs.get_data(picks=eeg_channels)  # (n_trials, n_ch, n_times)
        y = raw._epochs.events[:, 2]
        if scale:
            mean = X.mean(axis=-1, keepdims=True)
            std  = X.std(axis=-1, keepdims=True) + 1e-8
            X = (X - mean) / std
        X = X[:, np.newaxis, :, :].astype(np.float32)
        print(f"[Tensor] shape: {X.shape}")
        return X, y
    return extract


def extract_cwt(freqs: np.ndarray = None, n_cycles: float = 6.0):
    def extract(raw: mne.io.Raw):
        assert hasattr(raw, '_epochs'), "Add epoch() to the pipeline before extracting features."
        eeg_channels = _get_eeg_channels(raw)
        _freqs = freqs if freqs is not None else np.arange(1, 101, 1.0)
        raw_X = raw._epochs.get_data(picks=eeg_channels)  # (n_trials, n_ch, n_times)
        y = raw._epochs.events[:, 2]
        sfreq = raw.info['sfreq']

        # mne.time_frequency.tfr_array_morlet expects (n_trials, n_ch, n_times)
        # returns (n_trials, n_ch, n_freqs, n_times)
        X_cwt = mne.time_frequency.tfr_array_morlet(
            raw_X,
            sfreq=sfreq,
            freqs=_freqs,
            n_cycles=n_cycles,
            output='power'
        ).astype(np.float32)

        print(f"[CWT] shape: {X_cwt.shape}")
        return X_cwt, y
    return extract


def extract_bandpower(bands: dict = None):
    """Per-band log power → lightweight spectral baseline.

    Output shape: (n_trials, n_channels * n_bands) — float32.
    """
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
    return extract


# =============================================================================
# Pipeline
# =============================================================================

class MNEPipeline:
    """Composable EEG preprocessing pipeline.

    Steps are callables (Raw -> Raw) added via .add(). They execute in order
    when .run() is called. Feature extraction happens at the end via a separate
    extractor callable (Raw -> (X, y)).

    Example
    -------
    pipeline = MNEPipeline('subject01.fif')
    pipeline.add(notch(60))
    pipeline.add(bandpass([(8, 13), (13, 30)]))
    pipeline.add(rereference())
    pipeline.add(epoch(tmin=0, tmax=3))
    X, y = pipeline.run(extract_csp(n_components=6))
    """

    def __init__(self, fif_file: str):
        self.fif_file = fif_file
        self.steps = []

    def add(self, step) -> 'MNEPipeline':
        """Add a preprocessing step. Returns self for optional chaining."""
        self.steps.append(step)
        return self

    def describe(self):
        """Print the current pipeline steps in order."""
        print(f"Pipeline: {self.fif_file}")
        for i, step in enumerate(self.steps):
            print(f"  {i+1}. {repr(step)}")

    def run(self, extractor=None):
        """Execute all steps in order, then optionally extract features.

        Parameters
        ----------
        extractor : callable (Raw -> (X, y)), optional
            One of: extract_csp(), extract_tensor(), extract_cwt(),
            extract_bandpower(). If None, returns the processed Raw object.

        Returns
        -------
        (X, y) if extractor is provided, else processed mne.io.Raw
        """
        raw = mne.io.read_raw_fif(self.fif_file, preload=True)

        for step in self.steps:
            print(f"Applying: {repr(step)}")
            raw = step(raw) # step IS apply

        if extractor is not None:
            return extractor(raw)
        return raw


# =============================================================================
# Named presets — ready-to-use pipelines per paradigm
# =============================================================================

def preset_csp_lda(fif_file: str):
    """CSP + LDA — narrow µ/β bandpass, CSP features."""
    return (MNEPipeline(fif_file)
            .add(notch(60))
            .add(bandpass([(8, 30)]))
            .add(rereference())
            .add(epoch(tmin=0, tmax=3))
            .run(extract_csp(n_components=6)))


def preset_csp_lda_multiband(fif_file: str):
    """CSP + LDA — separate alpha and beta bands, concatenated CSP features."""
    return (MNEPipeline(fif_file)
            .add(notch(60))
            .add(bandpass([(8, 13), (13, 30)]))
            .add(rereference())
            .add(epoch(tmin=0, tmax=3))
            .run(extract_csp(n_components=6)))


def preset_deep_learning(fif_file: str):
    """EEGNet / ATCNet / CTNet / Conformer — wide bandpass, raw tensor."""
    return (MNEPipeline(fif_file)
            .add(notch(60))
            .add(bandpass([(4, 100)]))
            .add(rereference())
            .add(epoch(tmin=0, tmax=3))
            .run(extract_tensor(scale=True)))


def preset_cwt_hybrid(fif_file: str):
    """EMD+CWT+ADBN style — wide bandpass, CWT scalogram."""
    return (MNEPipeline(fif_file)
            .add(notch(60))
            .add(bandpass([(4, 100)]))
            .add(rereference())
            .add(epoch(tmin=0, tmax=3))
            .run(extract_cwt(freqs=np.arange(1, 101, 1.0))))




# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    EVENT_ID = {
    'move':       1,
    'jaw_clench': 2,
    }

    FILE = './data_collection/annotated_eeg/chengyi0210.fif'

    # -- Option A: use a preset
    X, y = preset_deep_learning(FILE)
    print(X.shape, np.unique(y))

    # -- Option B: build a custom pipeline
    pipeline = MNEPipeline(FILE)
    pipeline.add(notch(60))
    pipeline.add(plot_psd("Raw"))          # inspect before filtering
    pipeline.add(bandpass([(8, 13), (13, 30)]))
    pipeline.add(plot_psd("After filter")) # inspect after
    pipeline.add(rereference())
    pipeline.add(epoch(tmin=0, tmax=3, event_id=EVENT_ID))
    pipeline.describe()
    X, y = pipeline.run(extract_csp(n_components=6))
    print(X.shape, np.unique(y))