"""
Offline validation of the calibration pipeline using a real annotated FIF.

Splits the file into two halves:
  - First N_CALIB annotations → calibration (MI fine-tuning + jaw-clench baseline)
  - Remaining annotations     → validation

Validation scoring:
  move       — correct if the calibrated DeepConvNet predicts class 1 in any 3s
               sliding window within 4s of the event onset (stride = 1s)
  jaw_clench — correct if detect() fires on the 4s window

Usage:
  python ml_model/validate_calibration.py
  python ml_model/validate_calibration.py --n-calib 10 --unfreeze-blocks 1
  python ml_model/validate_calibration.py --fif data_collection/annotated_fifs/other.fif
"""

import os
import sys
import pickle
import argparse
import tempfile
import warnings
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ml_model import models
import ml_model.MI_calibration as _mi_calib
from ml_model.MI_calibration import run_preprocessing, run_finetuning, LABEL_MAP
from ml_model.jaw_clench_calibration import build_baseline, detect, WINDOW_SEC
from scipy.signal import sosfilt
from processing.preprocess_pipeline import add_idle_class

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_FIF      = os.path.join(ROOT, "data_collection", "annotated_fifs", "chengyi_4_21_1.fif")
DEFAULT_JAW_SVM  = os.path.join(ROOT, "ml_model", "loso_classical_models", "jaw_clench", "SVM_linear__alpha__bandpower.pkl")
DL_MODEL_PATH    = os.path.join(ROOT, "ml_model", "loso_dl_models", "move", "DeepConvNet_per_epoch.pt")

# Patch MI_calibration so run_finetuning also starts from the LOSO model
_mi_calib.MODEL_PATH = DL_MODEL_PATH
EEG_EXCLUDE  = {"Trigger", "Event"}
SFREQ        = 300
N_TIMEPOINTS = int(SFREQ * 3.0) + 1     # 901 samples — tmax=3.0 inclusive in MNE
VAL_WINDOW   = 4.0                       # seconds of EEG to check per validation event

_BP_BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
             "gamma": (30, 55), "emg": (65, 100)}


# ── Per-epoch preprocessing (matches classifier_worker.preprocess_epoch) ───────


def _preprocess_epoch(data: np.ndarray, ch_names: list) -> torch.Tensor:
    """Notch 60Hz → beta bandpass → average rereference → z-score.

    data     : (n_ch, n_samples) raw EEG
    Returns  : (1, 1, n_ch, 901) float32 tensor
    """
    data = data[:, -N_TIMEPOINTS:]
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.notch_filter(60.0, verbose=False)
    raw.filter(13, 30.0, verbose=False)
    raw.set_eeg_reference(ref_channels="average", verbose=False)
    X    = raw.get_data()
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-8
    X    = (X - mean) / std
    return torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0)


# ── Jaw-clench SVM feature extraction ────────────────────────────────────────


def _jaw_svm_features(data: np.ndarray, ch_names: list) -> np.ndarray:
    """Notch 60Hz → beta bandpass → average rereference → bandpower features.

    Matches classifier_worker.filter_for_svm + extract_bandpower_features:
    notch 60Hz, bandpass [13-30 Hz], avg rereference, then 5 bands × n_channels
    log-mean power. Returns (1, n_features) float32.
    """
    data  = data[:, -N_TIMEPOINTS:]
    info  = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arr = mne.filter.notch_filter(
            data[np.newaxis].astype(np.float64), Fs=SFREQ, freqs=60.0, verbose=False
        )
        epoch = mne.EpochsArray(arr, info,
                                events=np.array([[0, 0, 1]]), tmin=0, verbose=False)
        epoch.filter(13.0, 30.0, verbose=False)
        epoch.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    filtered = epoch.get_data()[0]   # (n_ch, n_times)
    feats = []
    for c in range(filtered.shape[0]):
        psd, freqs = mne.time_frequency.psd_array_welch(
            filtered[c][np.newaxis], sfreq=SFREQ,
            fmin=1, fmax=150, n_fft=min(filtered.shape[1], 256), verbose=False,
        )
        for lo, hi in _BP_BANDS.values():
            idx = (freqs >= lo) & (freqs <= hi)
            feats.append(float(np.log(psd[0, idx].mean() + 1e-10)))
    return np.array(feats, dtype=np.float32).reshape(1, -1)


# ── Idle window finder ────────────────────────────────────────────────────────


def _find_idle_onsets(all_anns: list, start_t: float, end_t: float,
                      window_dur: float, stride: float) -> list:
    """Return onset times for non-overlapping windows that don't touch any annotation."""
    busy = [(a["onset"], a["onset"] + float(a["duration"]) + window_dur)
            for a in all_anns]
    onsets = []
    t = start_t
    while t + window_dur <= end_t:
        if not any(not (t + window_dur <= b0 or t >= b1) for b0, b1 in busy):
            onsets.append(t)
        t += stride
    return onsets


# ── Calibration FIF builder ────────────────────────────────────────────────────


def _build_calibration_fif(raw: mne.io.Raw, calib_anns: list, tmp_dir: str) -> str:
    """Crop raw to the first n_calib annotations, add idle windows, save FIF."""
    last  = calib_anns[-1]
    end_t = last["onset"] + max(float(last["duration"]), 3.0) + VAL_WINDOW
    end_t = min(end_t, raw.times[-1])

    sub = raw.copy().crop(tmin=0.0, tmax=end_t)
    sub.set_annotations(mne.Annotations(
        onset=[a["onset"] for a in calib_anns],
        duration=[a["duration"] for a in calib_anns],
        description=[a["description"] for a in calib_anns],
    ))

    # Add idle epochs from unannotated gaps (start_min=0 → search from t=0)
    sub = add_idle_class(window_dur=3.0, idle_start_min=0.0)(sub)

    fif_path = os.path.join(tmp_dir, "calib_crop.fif")
    sub.save(fif_path, overwrite=True)
    return fif_path


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Offline calibration validation")
    p.add_argument("--n-calib",         type=int,   default=10,
                   help="Annotations used for calibration (default: 10)")
    p.add_argument("--base-model",      action="store_true",
                   help="Skip fine-tuning; validate with the base pretrained model only")
    p.add_argument("--unfreeze-blocks", type=int,   default=0, choices=range(5),
                   help="Layers to unfreeze for fine-tuning (default: 0=classifier only)")
    p.add_argument("--train-epochs",    type=int,   default=20,
                   help="Fine-tuning epochs (default: 20)")
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--weight-decay",    type=float, default=0.0,
                   help="L2 weight decay for Adam (default: 0.0)")
    p.add_argument("--stride",          type=float, default=1.0,
                   help="Sliding window stride in seconds (default: 1.0)")
    p.add_argument("--concurrent",      type=int,   default=1,
                   help="Consecutive correct predictions required (default: 1)")
    p.add_argument("--ratio-hits",      type=int,   default=None,
                   help="Hits needed in a rolling window to count as detected (e.g. 3)")
    p.add_argument("--ratio-window",    type=int,   default=4,
                   help="Rolling window size for --ratio-hits (default: 4)")
    p.add_argument("--svm-jaw-clench",  action="store_true",
                   help="Use SVM classifier for jaw clench instead of threshold detector")
    p.add_argument("--jaw-svm-path",   type=str,   default=DEFAULT_JAW_SVM,
                   help="Path to jaw-clench SVM .pkl (default: SVM_linear__alpha__bandpower.pkl)")
    p.add_argument("--fif",             type=str,   default=DEFAULT_FIF,
                   help="Path to annotated FIF file")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    print("=" * 60)
    print("  Offline Calibration Validation")
    print(f"  FIF:             {args.fif}")
    if args.base_model:
        print("  Mode:            BASE MODEL (no fine-tuning)")
    else:
        print(f"  Calibration:     first {args.n_calib} annotations")
        print(f"  Unfreeze blocks: {args.unfreeze_blocks}")
        print(f"  Train epochs:    {args.train_epochs}  lr={args.lr}")
    print(f"  Stride:          {args.stride}s")
    print(f"  Concurrent:      {args.concurrent}")
    print("=" * 60)

    raw     = mne.io.read_raw_fif(args.fif, preload=True, verbose=False)
    sfreq   = raw.info["sfreq"]
    eeg_chs = [ch for ch in raw.ch_names if ch not in EEG_EXCLUDE]
    all_anns = sorted(list(raw.annotations), key=lambda a: a["onset"])

    print(f"\n[validate] sfreq={sfreq:.0f}Hz  eeg_channels={eeg_chs}")
    print(f"[validate] Total annotations: {len(all_anns)}")

    if len(all_anns) <= args.n_calib:
        raise ValueError(
            f"Need more than {args.n_calib} annotations to have a validation set; "
            f"found {len(all_anns)}."
        )

    calib_anns = all_anns[:args.n_calib]
    val_anns   = all_anns[args.n_calib:]

    def _label_counts(anns):
        c = {}
        for a in anns:
            c[a["description"]] = c.get(a["description"], 0) + 1
        return c

    print(f"[validate] Calibration annotations: {_label_counts(calib_anns)}")
    print(f"[validate] Validation  annotations: {_label_counts(val_anns)}")

    with tempfile.TemporaryDirectory() as tmp_dir:

        # ── Step 1: Build calibration FIF ─────────────────────────────────────
        print("\n[Step 1] Building calibration FIF crop...")
        calib_fif = _build_calibration_fif(raw, calib_anns, tmp_dir)

        # ── Step 2: Fine-tune MI model (or load base) ─────────────────────────
        if args.base_model:
            print("\n[Step 2] Skipping fine-tuning — using base pretrained model.")
            used_model_path = DL_MODEL_PATH
        else:
            print("\n[Step 2] Preprocessing + fine-tuning MI model...")
            ft_args    = SimpleNamespace(
                unfreeze_blocks=args.unfreeze_blocks,
                train_epochs=args.train_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            X, y            = run_preprocessing(calib_fif)
            used_model_path = run_finetuning(X, y, ft_args)

        # ── Step 3: Build jaw-clench baseline ─────────────────────────────────
        print("\n[Step 3] Building jaw-clench baseline from idle segments...")
        baseline = build_baseline(calib_fif, label="idle")

        # ── Step 4: Load model ────────────────────────────────────────────────
        model = models.DeepConvNet(n_channels=len(eeg_chs), n_classes=2)
        model.load_state_dict(
            torch.load(used_model_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        print(f"[validate] Loaded model: {os.path.basename(used_model_path)}")

        # ── Step 4b: Load jaw-clench SVM (optional) ───────────────────────────
        jaw_svm = None
        if args.svm_jaw_clench:
            with open(args.jaw_svm_path, "rb") as f:
                jaw_svm = pickle.load(f)
            print(f"[validate] Loaded jaw SVM: {os.path.basename(args.jaw_svm_path)}")

        # ── Step 5: Validation loop ────────────────────────────────────────────
        tracked = {"move", "jaw_clench"}
        results = {lbl: {"correct": 0, "total": 0} for lbl in tracked}
        results["idle"] = {"correct": 0, "total": 0}

        stride_samps = int(args.stride * sfreq)
        move_class   = LABEL_MAP["move"]   # 1

        print(f"\n[Step 5] Validating {sum(1 for a in val_anns if a['description'] in tracked)} "
              f"tracked events...\n")

        for ann in val_anns:
            label = ann["description"]
            if label not in tracked:
                continue

            onset   = ann["onset"]
            start_s = int(onset * sfreq)
            end_s   = min(int((onset + VAL_WINDOW) * sfreq), raw.n_times)
            data    = raw.get_data(picks=eeg_chs, start=start_s, stop=end_s)

            results[label]["total"] += 1
            correct = False

            if label == "move":
                n_samps     = data.shape[1]
                consecutive = 0
                ring        = deque(maxlen=args.ratio_window)
                confs       = []
                for w0 in range(0, n_samps - N_TIMEPOINTS + 1, stride_samps):
                    window = data[:, w0: w0 + N_TIMEPOINTS]
                    x      = _preprocess_epoch(window, eeg_chs)
                    with torch.no_grad():
                        out  = model(x)
                        conf = out[0, move_class].item()
                        hit  = out.argmax(1).item() == move_class
                    confs.append(conf)
                    consecutive = consecutive + 1 if hit else 0
                    ring.append(int(hit))
                    if consecutive >= args.concurrent:
                        correct = True
                        break
                    if args.ratio_hits and len(ring) == args.ratio_window and sum(ring) >= args.ratio_hits:
                        correct = True
                        break
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                max_conf = max(confs) if confs else 0.0

            elif label == "jaw_clench":
                win_samps   = int(WINDOW_SEC * sfreq) if jaw_svm is None else N_TIMEPOINTS
                consecutive = 0
                ring        = deque(maxlen=args.ratio_window)
                confs       = []
                for w0 in range(0, data.shape[1] - win_samps + 1, stride_samps):
                    window = data[:, w0: w0 + win_samps]
                    if jaw_svm is not None:
                        feats = _jaw_svm_features(window, eeg_chs)
                        hit   = jaw_svm.predict(feats)[0] == 1
                        conf  = jaw_svm.predict_proba(feats)[0][1]
                    else:
                        filtered = sosfilt(baseline['sos'], window, axis=1)
                        rms  = float(np.sqrt((filtered ** 2).mean()))
                        conf = rms / baseline['threshold']   # >1 means detected
                        hit  = rms > baseline['threshold']
                    confs.append(conf)
                    consecutive = consecutive + 1 if hit else 0
                    ring.append(int(hit))
                    if consecutive >= args.concurrent:
                        correct = True
                        break
                    if args.ratio_hits and len(ring) == args.ratio_window and sum(ring) >= args.ratio_hits:
                        correct = True
                        break
                avg_conf = sum(confs) / len(confs) if confs else 0.0

            results[label]["correct"] += int(correct)
            mark = "✓" if correct else "✗"
            conf_str = f"  avg={avg_conf:.2f}  max={max_conf:.2f}" if label == "move" else f"  avg_conf={avg_conf:.2f}"
            print(f"  {mark}  {label:<14}  onset={onset:7.1f}s  {conf_str}")

        # ── Idle (true-negative) check ────────────────────────────────────────
        val_start = val_anns[0]["onset"]
        idle_onsets = _find_idle_onsets(
            all_anns, val_start, raw.times[-1],
            window_dur=3.0, stride=args.stride,
        )
        win_samps_jc = int(WINDOW_SEC * sfreq)
        print(f"\n  Checking {len(idle_onsets)} idle windows...\n")
        for onset in idle_onsets:
            start_s = int(onset * sfreq)
            end_s   = min(int((onset + 3.0) * sfreq) + 1, raw.n_times)
            data    = raw.get_data(picks=eeg_chs, start=start_s, stop=end_s)

            x = _preprocess_epoch(data, eeg_chs)
            with torch.no_grad():
                pred = model(x).argmax(1).item()
            jc = detect(data[:, :win_samps_jc], baseline)

            correct = (pred != move_class) and (not jc)
            results["idle"]["total"]   += 1
            results["idle"]["correct"] += int(correct)
            mark = "✓" if correct else "✗"
            print(f"  {mark}  idle            onset={onset:7.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    total_c, total_t = 0, 0
    for label in sorted(results):
        c   = results[label]["correct"]
        t   = results[label]["total"]
        pct = 100 * c / t if t else 0.0
        print(f"  {label:<16}: {c:2d}/{t:<2d}  ({pct:.1f}%)")
        total_c += c
        total_t += t
    overall = 100 * total_c / total_t if total_t else 0.0
    print(f"  {'OVERALL':<16}: {total_c:2d}/{total_t:<2d}  ({overall:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
