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
import argparse
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ml_model import models
from ml_model.MI_calibration import run_preprocessing, run_finetuning, LABEL_MAP, MODEL_PATH
from ml_model.jaw_clench_calibration import build_baseline, detect, WINDOW_SEC
from processing.preprocess_pipeline import add_idle_class

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_FIF  = os.path.join(ROOT, "data_collection", "annotated_fifs", "chengyi_4_21_1.fif")
EEG_EXCLUDE  = {"Trigger", "Event"}
SFREQ        = 300
N_TIMEPOINTS = int(SFREQ * 3.0) + 1     # 901 samples — tmax=3.0 inclusive in MNE
VAL_WINDOW   = 4.0                       # seconds of EEG to check per validation event
STRIDE_SEC   = 1.0                       # stride for sliding inference windows


# ── Per-epoch preprocessing (matches classifier_worker.preprocess_epoch) ───────


def _preprocess_epoch(data: np.ndarray, ch_names: list) -> torch.Tensor:
    """Notch 60Hz → beta bandpass → average rereference → z-score.

    data     : (n_ch, n_samples) raw EEG
    Returns  : (1, 1, n_ch, 901) float32 tensor
    """
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.notch_filter(60.0, verbose=False)
    raw.filter(13, 30.0, verbose=False)
    raw.set_eeg_reference(ref_channels="average", verbose=False)
    X    = raw.get_data()[:, -N_TIMEPOINTS:]
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-8
    X    = (X - mean) / std
    return torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0)


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
            used_model_path = MODEL_PATH
        else:
            print("\n[Step 2] Preprocessing + fine-tuning MI model...")
            ft_args    = SimpleNamespace(
                unfreeze_blocks=args.unfreeze_blocks,
                train_epochs=args.train_epochs,
                lr=args.lr,
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

        # ── Step 5: Validation loop ────────────────────────────────────────────
        tracked = {"move", "jaw_clench"}
        results = {lbl: {"correct": 0, "total": 0} for lbl in tracked}

        stride_samps = int(STRIDE_SEC * sfreq)
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
                # Slide a 3s window (1s stride) across the 4s event window.
                # Correct if model predicts class 1 at any position.
                n_samps = data.shape[1]
                for w0 in range(0, n_samps - N_TIMEPOINTS + 1, stride_samps):
                    window = data[:, w0: w0 + N_TIMEPOINTS]
                    x      = _preprocess_epoch(window, eeg_chs)
                    with torch.no_grad():
                        out  = model(x)
                        pred = out.argmax(1).item()
                    if pred == move_class:
                        correct = True
                        break

            elif label == "jaw_clench":
                # Slide a window matching the size used to build the threshold.
                # Passing the full 4s window dilutes clench energy and misses detections.
                win_samps = int(WINDOW_SEC * sfreq)
                for w0 in range(0, data.shape[1] - win_samps + 1, win_samps):
                    if detect(data[:, w0: w0 + win_samps], baseline):
                        correct = True
                        break

            results[label]["correct"] += int(correct)
            mark = "✓" if correct else "✗"
            print(f"  {mark}  {label:<14}  onset={onset:7.1f}s")

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
