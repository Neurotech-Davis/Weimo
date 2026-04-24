"""
Offline batch evaluation — confusion matrix.

For each ground-truth annotation, the model is given a 4-second window after
the annotation onset. If it predicts the correct class at least once in that
window, it's a success. Otherwise the most common wrong prediction is recorded.

Run from ml_model/:
    python offline_eval.py
"""

import os
import sys
import warnings
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model import models
from processing.preprocess_pipeline import add_idle_class

# =============================================================================
# CONFIG — edit these
# =============================================================================

FIF_PATH   = "../data_collection/annotated_fifs/chengyi_4_21_1.fif"
MODEL_PATH = "./loso_dl_models/binary/DeepConvNet_per_epoch.pt"

LABEL_MAP     = {"idle": 0, "move": 1, "jaw_clench": 0}
CLASS_NAMES   = ["idle", "move", "jaw_clench"]
DETECTION_SEC = 4.0     # window after annotation onset to look for correct prediction

N_CHANNELS   = 8
N_CLASSES    = 2
SFREQ        = 300
TRIAL_DUR    = 3.0
STRIDE_SEC   = 1.0
N_TIMEPOINTS = int(SFREQ * TRIAL_DUR) + 1   # 901

BANDPASS = (13, 30)
NOTCH    = 60.0

# =============================================================================
# Preprocessing
# =============================================================================

CH_NAMES = [
    "EEG LE-Pz", "EEG F4-Pz", "EEG C4-Pz", "EEG P4-Pz",
    "EEG P3-Pz", "EEG C3-Pz", "EEG F3-Pz", "Pz",
]

def preprocess_window(data: np.ndarray) -> torch.Tensor:
    """Replicates epoch_filter(bands=[(13,30)], notch_freq=60, ref='average') + extract_tensor(scale=True).

    Matches the per-epoch preprocessing used during training exactly.
    """
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types="eeg")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # 1. notch — mne.filter.notch_filter on 3D array, same as epoch_filter
        arr = mne.filter.notch_filter(
            data[np.newaxis],  # (1, 8, 901)
            Fs=SFREQ, freqs=NOTCH, verbose=False,
        )

        # 2. bandpass — epochs.filter(), same as epoch_filter
        events = np.array([[0, 0, 1]])
        epoch  = mne.EpochsArray(arr, info, events=events, tmin=0, verbose=False)
        epoch.filter(BANDPASS[0], BANDPASS[1], verbose=False)

        # 3. rereference — epochs.set_eeg_reference(projection=False), same as epoch_filter
        epoch.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

    X    = epoch.get_data()[0]  # (8, 901)
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-8
    X    = (X - mean) / std
    return torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0)


# =============================================================================
# Load model
# =============================================================================

def load_model():
    m = models.DeepConvNet(N_CHANNELS, N_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    print(f"Loaded model: {MODEL_PATH}")
    return m


# =============================================================================
# Load file
# =============================================================================

def load_file(fif_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    duration = raw.times[-1]
    eeg_chs  = [ch for ch in raw.ch_names if ch not in {"Trigger", "Event"}]
    data     = raw.get_data(picks=eeg_chs)
    print(f"File: {os.path.basename(fif_path)}  |  {duration:.1f}s  |  {len(eeg_chs)} channels")

    # add idle windows from unannotated time — same as training pipeline
    raw = add_idle_class(window_dur=TRIAL_DUR, idle_start_min=1.0)(raw)

    gt_events = [
        (ann["onset"], ann["onset"] + ann["duration"], ann["description"])
        for ann in raw.annotations
        if ann["description"] in LABEL_MAP
    ]
    counts = Counter(desc for _, _, desc in gt_events)
    print(f"Annotations: {len(gt_events)}  |  {dict(counts)}")
    return data, duration, gt_events


# =============================================================================
# Sliding window inference
# =============================================================================

def run_inference(data, model):
    stride_samples = int(SFREQ * STRIDE_SEC)
    window_samples = N_TIMEPOINTS
    n_samples      = data.shape[1]

    results = []   # (t_start, pred, conf)
    start = 0
    while start + window_samples <= n_samples:
        window = data[:, start : start + window_samples]
        t_start = start / SFREQ

        x = preprocess_window(window)
        with torch.no_grad():
            out  = model(x)
            pred = out.argmax(1).item()
            conf = out[0, pred].item()

        results.append((t_start, pred, conf))
        start += stride_samples

    print(f"Inference complete: {len(results)} windows")
    return results


# =============================================================================
# Evaluate — confusion matrix
# =============================================================================

def evaluate(results, gt_events):
    """
    For each annotation, look at all predictions in [onset, onset + DETECTION_SEC].
    Success: correct class predicted at least once → predicted = true label.
    Failure: correct class never predicted → predicted = most common class in window.
    """
    true_labels = []
    pred_labels = []

    for onset, offset, label in gt_events:
        true_cls = LABEL_MAP[label]
        window_end = onset + DETECTION_SEC

        preds_in_window = [
            pred for t, pred, conf in results
            if onset <= t <= window_end
        ]

        if not preds_in_window:
            # no inference windows fell in this range — skip
            continue

        if true_cls in preds_in_window:
            assigned_pred = true_cls   # success
        else:
            assigned_pred = Counter(preds_in_window).most_common(1)[0][0]

        true_labels.append(true_cls)
        pred_labels.append(assigned_pred)

    return np.array(true_labels), np.array(pred_labels)


# =============================================================================
# Plot confusion matrix
# =============================================================================

def plot_confusion_matrix(true_labels, pred_labels):
    from sklearn.metrics import confusion_matrix, classification_report

    present = sorted(set(true_labels) | set(pred_labels))
    names   = [CLASS_NAMES[i] for i in present]
    cm      = confusion_matrix(true_labels, pred_labels, labels=present)

    acc = (true_labels == pred_labels).mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Offline eval — {os.path.basename(FIF_PATH)}\n"
        f"Detection window: {DETECTION_SEC}s after onset  |  "
        f"overall: {acc:.2%}  ({(true_labels == pred_labels).sum()}/{len(true_labels)})"
    )
    plt.tight_layout()

    print("\n" + classification_report(true_labels, pred_labels,
                                       labels=present, target_names=names,
                                       zero_division=0))
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    model              = load_model()
    data, duration, gt = load_file(FIF_PATH)
    results            = run_inference(data, model)
    true_labels, pred_labels = evaluate(results, gt)
    plot_confusion_matrix(true_labels, pred_labels)
