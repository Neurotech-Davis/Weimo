"""
Jaw clench detector — core module.

Provides:
  build_baseline(fif_path) -> dict
      Compute a detection threshold from a .fif file.
      If the file has annotations, pass label='idle' to use only rest segments
      (e.g. from a calibration FIF produced by ml_calibration.py).

  make_filter_state(baseline, n_channels) -> ndarray
      Create initial filter state for stateful real-time calls.

  detect(window, baseline, zi=None) -> bool | (bool, zi_new)
      Given a raw EEG window (n_channels, n_samples), return True if a jaw
      clench is detected.

      Pass zi (from make_filter_state or previous detect call) to maintain
      filter state across consecutive windows — required for correct real-time
      operation. Returns (bool, zi_new) when zi is supplied, plain bool otherwise.

  save_baseline(baseline, path) / load_baseline(path)
      Persist/restore the threshold and sfreq to/from a JSON file.

  get_latest_calibration_fif(calib_dir) -> str | None
      Return the most recently saved *_calibration.fif in a directory.

Import this module anywhere (live worker, validation script, notebook).
All parameters are in the baseline dict so callers don't need to know internals.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import mne
from scipy.signal import butter, sosfilt, sosfilt_zi

# ── Parameters ────────────────────────────────────────────────────────────────
EMG_LOW      = 65.0   # Hz — lower edge of EMG band
EMG_HIGH     = 100.0  # Hz — upper edge
WINDOW_SEC   = 0.1    # RMS window length (seconds)
THRESHOLD_K  = 4.0    # threshold = baseline_mean + K * baseline_std
BASELINE_SEC = 60.0   # seconds of .fif to use when no label filter is applied

EXCLUDE = {"Trigger", "Event", "TRG"}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALIB_FIF_DIR    = os.path.join(ROOT, "data_collection", "calibration_fifs")
DEFAULT_SAVE_PATH = os.path.join(ROOT, "src", "models", "jaw_clench_baseline.json")


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


def get_latest_calibration_fif(calib_dir: str = CALIB_FIF_DIR) -> "str | None":
    """Return the most recently saved *_calibration.fif in calib_dir, or None."""
    matches = glob.glob(os.path.join(calib_dir, "*_calibration.fif"))
    return max(matches, key=os.path.getmtime) if matches else None


def save_baseline(baseline: dict, path: str = DEFAULT_SAVE_PATH):
    """Persist threshold + sfreq to JSON (sos is recomputed from sfreq on load)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {k: baseline[k] for k in ("threshold", "sfreq", "mean", "std")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[jaw_clench] Saved baseline → {path}")


def load_baseline(path: str = DEFAULT_SAVE_PATH) -> dict:
    """Load a previously saved baseline and reconstruct the filter."""
    with open(path) as f:
        payload = json.load(f)
    payload["sos"] = _make_filter(payload["sfreq"])
    return payload


def build_baseline(fif_path: str,
                   label: "str | None" = None,
                   baseline_sec: float = BASELINE_SEC,
                   threshold_k: float  = THRESHOLD_K) -> dict:
    """
    Load a .fif file and compute a jaw-clench detection threshold.

    If `label` is given and the file has matching annotations (e.g. 'idle'
    from a calibration FIF), only those labeled segments are used — ideal
    for files from ml_calibration.py where rest trials are annotated 'idle'.
    Falls back to the first `baseline_sec` seconds when no label is provided
    or no matching annotations exist.

    Parameters
    ----------
    fif_path     : path to any .fif recording (annotated or raw)
    label        : annotation label to extract (e.g. 'idle'); None = use whole file
    baseline_sec : seconds to use when label filtering is not applied
    threshold_k  : threshold = mean + k * std

    Returns
    -------
    dict with keys:
      'threshold' : float   — RMS value above which a jaw clench is flagged
      'sos'       : ndarray — IIR filter coefficients
      'sfreq'     : float   — sample rate of the source file
      'mean'      : float   — baseline mean RMS (for diagnostics)
      'std'       : float   — baseline std RMS (for diagnostics)
    """
    raw    = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    sfreq  = raw.info["sfreq"]
    eeg_ch = [ch for ch in raw.ch_names if ch not in EXCLUDE]
    sos    = _make_filter(sfreq)
    win    = int(WINDOW_SEC * sfreq)

    # Extract segments for the baseline
    segments = []
    if label is not None and len(raw.annotations) > 0:
        matching = [a for a in raw.annotations if a["description"] == label]
        for ann in matching:
            start = int(ann["onset"] * sfreq)
            stop  = int((ann["onset"] + ann["duration"]) * sfreq)
            seg   = raw.get_data(picks=eeg_ch, start=start, stop=stop)
            if seg.shape[1] >= win:
                segments.append(seg)

    if segments:
        data_source = f"{len(segments)} '{label}' segments"
        data = np.concatenate(segments, axis=1)
    else:
        if label is not None:
            print(f"[jaw_clench] No '{label}' annotations found — using first {baseline_sec}s")
        n_samples = int(baseline_sec * sfreq)
        data = raw.get_data(picks=eeg_ch)[:, :n_samples]
        data_source = f"first {baseline_sec}s"

    # Filter whole concatenated segment first, then compute windowed RMS —
    # avoids FIR transients that appear when filtering each short window in isolation
    filtered = sosfilt(sos, data, axis=1)
    n_win    = filtered.shape[1] // win
    rms_vals = np.array([
        float(np.sqrt((filtered[:, i * win:(i + 1) * win] ** 2).mean()))
        for i in range(n_win)
    ])

    mean      = float(rms_vals.mean())
    std       = float(rms_vals.std())
    threshold = mean + threshold_k * std

    print(f"[jaw_clench] Baseline from '{os.path.basename(fif_path)}' ({data_source}): "
          f"mean={mean:.3e}  std={std:.3e}  threshold={threshold:.3e}")

    return {"threshold": threshold, "sos": sos, "sfreq": sfreq, "mean": mean, "std": std}


def main():
    p = argparse.ArgumentParser(
        description="Compute jaw-clench baseline from a calibration FIF and save it."
    )
    p.add_argument("--fif-path",     type=str,   default=None,
                   help="Path to .fif file (default: latest in data_collection/calibration_fifs/)")
    p.add_argument("--label",        type=str,   default="idle",
                   help="Annotation label to use as quiet baseline (default: idle)")
    p.add_argument("--threshold-k",  type=float, default=THRESHOLD_K,
                   help=f"Threshold multiplier (default: {THRESHOLD_K})")
    p.add_argument("--baseline-sec", type=float, default=BASELINE_SEC,
                   help=f"Seconds to use when no label filter applies (default: {BASELINE_SEC})")
    p.add_argument("--save-path",    type=str,   default=DEFAULT_SAVE_PATH,
                   help=f"Where to save the baseline JSON (default: {DEFAULT_SAVE_PATH})")
    args = p.parse_args()

    fif_path = args.fif_path
    if fif_path is None:
        fif_path = get_latest_calibration_fif()
        if fif_path is None:
            print(f"[jaw_clench] No calibration FIFs found in {CALIB_FIF_DIR}")
            print("[jaw_clench] Run ml_calibration.py first, or pass --fif-path explicitly.")
            sys.exit(1)
        print(f"[jaw_clench] Using latest calibration FIF: {fif_path}")

    baseline = build_baseline(
        fif_path,
        label=args.label or None,
        baseline_sec=args.baseline_sec,
        threshold_k=args.threshold_k,
    )
    save_baseline(baseline, args.save_path)


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


if __name__ == "__main__":
    main()
