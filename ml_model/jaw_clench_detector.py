"""
Jaw clench detector — core module.

Provides:
  build_baseline(fif_path) -> dict
      Compute a detection threshold from a .fif file.

  make_filter_state(baseline, n_channels) -> ndarray
      Create initial filter state for stateful real-time calls.

  detect(window, baseline, zi=None) -> bool | (bool, zi_new)
      Given a raw EEG window (n_channels, n_samples), return True if a jaw
      clench is detected.

      Pass zi (from make_filter_state or previous detect call) to maintain
      filter state across consecutive windows — required for correct real-time
      operation. Returns (bool, zi_new) when zi is supplied, plain bool otherwise.

Import this module anywhere (live worker, validation script, notebook).
All parameters are in the baseline dict so callers don't need to know internals.
"""

import numpy as np
import mne
from scipy.signal import butter, sosfilt, sosfilt_zi

# ── Parameters ────────────────────────────────────────────────────────────────
EMG_LOW      = 65.0   # Hz — lower edge of EMG band
EMG_HIGH     = 100.0  # Hz — upper edge
WINDOW_SEC   = 0.1    # RMS window length (seconds)
THRESHOLD_K  = 4.0    # threshold = baseline_mean + K * baseline_std
BASELINE_SEC = 60.0   # seconds of .fif to use as quiet baseline

EXCLUDE = {"Trigger", "Event", "TRG"}


def _make_filter(sfreq: float):
    """Build a 4th-order Butterworth EMG bandpass filter."""
    nyq = sfreq / 2.0
    return butter(4, [EMG_LOW / nyq, EMG_HIGH / nyq], btype='bandpass', output='sos')


def make_filter_state(baseline: dict, n_channels: int) -> np.ndarray:
    """
    Create a zero-initialized filter state for stateful real-time detect() calls.

    Parameters
    ----------
    baseline   : dict returned by build_baseline()
    n_channels : number of EEG channels the window will have

    Returns
    -------
    zi : ndarray, shape (n_sections, n_channels, 2)
    """
    zi_1ch = sosfilt_zi(baseline['sos'])          # (n_sections, 2)
    return np.broadcast_to(
        zi_1ch[:, np.newaxis, :], (zi_1ch.shape[0], n_channels, 2)
    ).copy()


def build_baseline(fif_path: str,
                   baseline_sec: float = BASELINE_SEC,
                   threshold_k: float  = THRESHOLD_K) -> dict:
    """
    Load a .fif file, use the first `baseline_sec` seconds as a quiet
    reference, and return a baseline dict ready for use with detect().

    Parameters
    ----------
    fif_path     : path to any .fif recording (annotated or raw)
    baseline_sec : how many seconds to use (should be quiet — no clenching)
    threshold_k  : threshold = mean + k * std

    Returns
    -------
    dict with keys:
      'threshold' : float  — RMS value above which a jaw clench is flagged
      'sos'       : ndarray — IIR filter coefficients
      'sfreq'     : float  — sample rate of the source file
      'mean'      : float  — baseline mean RMS (for diagnostics)
      'std'       : float  — baseline std RMS (for diagnostics)
    """
    raw    = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq  = raw.info['sfreq']
    eeg_ch = [ch for ch in raw.ch_names if ch not in EXCLUDE]
    data   = raw.get_data(picks=eeg_ch)

    # trim to baseline window
    n_samples = int(baseline_sec * sfreq)
    data      = data[:, :n_samples]

    sos = _make_filter(sfreq)

    # filter the full baseline segment first, then window — avoids filter
    # transients that would occur if each short window were filtered in isolation
    filtered = sosfilt(sos, data, axis=1)

    win      = int(WINDOW_SEC * sfreq)
    n_win    = filtered.shape[1] // win
    rms_vals = np.array([
        float(np.sqrt((filtered[:, i*win:(i+1)*win] ** 2).mean()))
        for i in range(n_win)
    ])

    mean      = float(rms_vals.mean())
    std       = float(rms_vals.std())
    threshold = mean + threshold_k * std

    print(f"[jaw_clench] Baseline from '{fif_path}' ({baseline_sec}s): "
          f"mean={mean:.3e}  std={std:.3e}  threshold={threshold:.3e}")

    return {
        'threshold' : threshold,
        'sos'       : sos,
        'sfreq'     : sfreq,
        'mean'      : mean,
        'std'       : std,
    }


def detect(window: np.ndarray, baseline: dict, zi=None):
    """
    Detect whether a jaw clench is present in a single EEG window.

    Parameters
    ----------
    window   : np.ndarray, shape (n_channels, n_samples) — raw EEG, no preprocessing needed
    baseline : dict returned by build_baseline()
    zi       : filter state from make_filter_state() or a previous detect() call.
               When provided, filter state is maintained across consecutive windows,
               which is required for correct real-time operation.
               When None, filter state starts from zero each call — only accurate
               for windows long enough for the filter to settle (several hundred ms).

    Returns
    -------
    bool                  — if zi is None
    (bool, zi_new)        — if zi is provided; pass zi_new to the next detect() call
    """
    sos = baseline['sos']
    if zi is not None:
        filtered, zi_new = sosfilt(sos, window, axis=1, zi=zi)
        rms = float(np.sqrt((filtered ** 2).mean()))
        return rms > baseline['threshold'], zi_new
    else:
        filtered = sosfilt(sos, window, axis=1)
        rms = float(np.sqrt((filtered ** 2).mean()))
        return rms > baseline['threshold']
