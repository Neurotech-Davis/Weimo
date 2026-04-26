"""
For this worker to work, it has to be connected to the DSI-7s LSL streamer
- (initialize the streaming layer first)
"""

import time
import sys
import os
from collections import deque
import pickle
import warnings

import numpy as np
import torch
import mne

mne.set_log_level("ERROR")
# this silences the MNE warnings about filter length
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

from mne_lsl.lsl import local_clock
from mne_lsl.stream import StreamLSL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml_model import models

PREDICTION_WINDOW_LEN = 3
PREDICTION_THRESHOLD = 2
LABEL_MAP = {0: "idle", 1: "move", 2: "jaw_clench"}
EXCLUDE = {"Trigger", "Event"}

CONFIGS = {
    "N_CHANNELS": 8,
    "N_CLASSES": 2,  # 2 for binary
    "TRIAL_DUR": 3,
    "SFREQ": 300,
    "STRIDE_SEC": 0.5,
    "LEAD_IN": 2,
}
CONFIGS["N_TIMEPOINTS"] = (
    int(CONFIGS["SFREQ"] * CONFIGS["TRIAL_DUR"]) + 1
)  # MNE tmax inclusive → 901

_MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH     = os.path.join(_MODELS_DIR, "DeepConvNet_per_epoch.pt")
SVM_MODEL_PATH = os.path.join(_MODELS_DIR, "SVM_linear__beta__bandpower.pkl")
STREAM_NAME = "WS-default"

CH_NAMES = [
    "EEG LE-Pz",
    "EEG F4-Pz",
    "EEG C4-Pz",
    "EEG P4-Pz",
    "EEG P3-Pz",
    "EEG C3-Pz",
    "EEG F3-Pz",
    "Pz",
]

_BP_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 55),
    "emg": (65, 100),
}


# ── Preprocessing ─────────────────────────────────────────────────────────────


def preprocess_epoch(data: np.ndarray) -> torch.Tensor:
    """
    Replicates the offline training pipeline on a single (n_channels, n_timepoints) window.
    Channel order must match training:
      ['EEG LE-Pz', 'EEG F4-Pz', 'EEG C4-Pz', 'EEG P4-Pz',
       'EEG P3-Pz', 'EEG C3-Pz', 'EEG F3-Pz', 'Pz']
    Returns: (1, 1, 8, 901) float32 tensor
    """
    # data = data - data.mean(axis=-1, keepdims=True)
    # data[7, :] = 0.0

    ch_names = [
        "EEG LE-Pz",
        "EEG F4-Pz",
        "EEG C4-Pz",
        "EEG P4-Pz",
        "EEG P3-Pz",
        "EEG C3-Pz",
        "EEG F3-Pz",
        "Pz",
    ]
    data = data[:, -CONFIGS["N_TIMEPOINTS"]:]  # clip before filtering — matches epoch_filter in training
    info = mne.create_info(ch_names=ch_names, sfreq=CONFIGS["SFREQ"], ch_types="eeg")
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose=False)

    raw.notch_filter(60.0, verbose=False)
    raw.filter(13, 30.0, verbose=False)
    raw.set_eeg_reference(ref_channels="average", verbose=False)

    X = raw.get_data()
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mean) / std

    return torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 900)


def filter_for_svm(data: np.ndarray) -> np.ndarray:
    """Replicates epoch_filter(bands=[(13,30)], notch_freq=60, ref='average') — no z-score.

    Clips to N_TIMEPOINTS BEFORE filtering so the FIR filter sees the same 901-sample
    window it saw during training (not the full 5s TRIAL_DUR+LEAD_IN window).
    Returns (n_channels, N_TIMEPOINTS) float64 array ready for bandpower extraction.
    """
    data = data.astype(np.float64)[:, -CONFIGS["N_TIMEPOINTS"] :]  # clip to 901 first — matches training
    info = mne.create_info(ch_names=CH_NAMES, sfreq=CONFIGS["SFREQ"], ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arr = mne.filter.notch_filter(
            data[np.newaxis], Fs=CONFIGS["SFREQ"], freqs=60.0, verbose=False
        )
        epoch = mne.EpochsArray(
            arr, info, events=np.array([[0, 0, 1]]), tmin=0, verbose=False
        )
        epoch.filter(13, 30.0, verbose=False)
        epoch.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    return epoch.get_data()[0]


def extract_bandpower_features(filtered: np.ndarray) -> np.ndarray:
    """Bandpower features matching extract_bandpower() used during SVM training.

    Returns (1, 40) float32 array: 5 bands × 8 channels, log-mean power each.
    """
    feats = []
    for c in range(filtered.shape[0]):
        psd, freqs = mne.time_frequency.psd_array_welch(
            filtered[c][np.newaxis],
            sfreq=CONFIGS["SFREQ"],
            fmin=1,
            fmax=150,
            n_fft=min(filtered.shape[1], 256),
            verbose=False,
        )
        for lo, hi in _BP_BANDS.values():
            idx = (freqs >= lo) & (freqs <= hi)
            feats.append(float(np.log(psd[0, idx].mean() + 1e-10)))
    return np.array(feats, dtype=np.float32).reshape(1, -1)


# ── Signal quality check ──────────────────────────────────────────────────────


REFERENCE_CHANNELS = {"Pz", "TRG"}


def check_signal_quality(data: np.ndarray, eeg_picks: list) -> bool:
    """
    Prints per-channel stats and returns False if any channel is flatlined.
    """
    any_flat = False
    for i, ch in enumerate(eeg_picks):
        ch_std = data[i].std()
        ch_max = data[i].max()
        ch_min = data[i].min()
        flat = ch_std < 1e-6 and ch not in REFERENCE_CHANNELS
        if flat:
            any_flat = True
            print("[classifier_worker] Warning: flat channels")
        # flag = "⚠ FLAT" if flat else ""
        # print(
        #     f"  {ch:<20} std={ch_std:.6f}  max={ch_max:.6f}  min={ch_min:.6f}  {flag}"
        # )
    return not any_flat


# ── LSL connection ────────────────────────────────────────────────────────────


def attempt_LSL_connection(max_retries: int, retry_delay: int):
    stream = None
    for attempt in range(1, max_retries + 1):
        try:
            stream = StreamLSL(
                bufsize=CONFIGS["TRIAL_DUR"] + CONFIGS["LEAD_IN"] + 1.0,
                name=STREAM_NAME,
            ).connect()
            print(f"[classifier_worker] Connected on attempt {attempt}.")
            break
        except Exception as e:
            print(
                f"[classifier_worker] Attempt {attempt}/{max_retries} failed: {e}. "
                f"Retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
    return stream


# ── Model ─────────────────────────────────────────────────────────────────────


def load_model():
    model = models.DeepConvNet(CONFIGS["N_CHANNELS"], CONFIGS["N_CLASSES"]).to("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"[classifier_worker] Loaded DL model from {MODEL_PATH}")
    return model


def load_svm():
    with open(SVM_MODEL_PATH, "rb") as f:
        svm = pickle.load(f)
    print(f"[classifier_worker] Loaded SVM from {SVM_MODEL_PATH}")
    return svm


# ── Worker ────────────────────────────────────────────────────────────────────


def classifier_worker(shared_state):
    stream = None
    try:
        # SETUP
        model = load_model()
        svm = load_svm()
        stream = attempt_LSL_connection(max_retries=10, retry_delay=2)
        if stream is None:
            raise RuntimeError(
                f"[classifier_worker] Could not connect to LSL stream '{STREAM_NAME}'. "
                f"Is dsi2lsl running?"
            )

        all_ch = stream.info["ch_names"]
        sfreq = stream.info["sfreq"]
        eeg_picks = [ch for ch in all_ch if ch not in EXCLUDE]

        print("\n--- Stream Info ---")
        print(f"  All Channels ({len(all_ch)}): {all_ch}")
        print(f"  Selected Channels ({len(eeg_picks)}): {eeg_picks}")
        print(f"  Sfreq: {sfreq} Hz")
        print(
            f"  Expected shape: ({len(eeg_picks)}, {int(sfreq * CONFIGS['TRIAL_DUR'])})"
        )
        print("-------------------\n")

        shared_state.classifier_running.value = True

        # let buffer fill before first read
        time.sleep(CONFIGS["TRIAL_DUR"] + 0.5)

        last_data = None

        # LOOP
        pred_history = deque(maxlen=PREDICTION_WINDOW_LEN)
        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            data, timestamps = stream.get_data(
                winsize=CONFIGS["TRIAL_DUR"] + CONFIGS["LEAD_IN"],
                picks=eeg_picks,
            )

            # freshness check
            stream_age = local_clock() - timestamps[-1]
            if stream_age > 0.5:
                print(
                    f"[classifier_worker] ⚠ Stream is {stream_age:.2f}s stale, skipping"
                )
                time.sleep(0.1)
                continue

            # identical window check
            if last_data is not None and np.array_equal(data, last_data):
                print("[classifier_worker] ⚠ Data identical to last window")
            last_data = data.copy()

            # signal quality — remove this block once hardware is validated
            # print("[classifier_worker] Signal quality:")
            signal_ok = check_signal_quality(data, eeg_picks)
            if not signal_ok:
                print(
                    "[classifier_worker] ⚠ Flatlined channel detected — skipping inference"
                )
                time.sleep(0.1)
                continue

            # ── DL inference (move) ───────────────────────────────────────────
            x = preprocess_epoch(data)
            if torch.isnan(x).any():
                print("[classifier_worker] ⚠ NaNs in tensor — skipping")
                continue

            with torch.no_grad():
                dl_out = model(x)
                dl_pred = dl_out.argmax(1).item()
                dl_conf = dl_out[0, dl_pred].item()

            # ── SVM inference (jaw_clench) ────────────────────────────────────
            filtered = filter_for_svm(data)
            feats = extract_bandpower_features(filtered)
            svm_pred_raw = svm.predict(feats)[0]  # 0=idle, 1=jaw_clench
            svm_conf = svm.predict_proba(feats)[0][svm_pred_raw]

            # ── Combine & decide ──────────────────────────────────────────────
            # raw argmax, no confidence gate — consensus window handles filtering
            if dl_pred == 1:
                final_pred = 1  # move
            elif svm_pred_raw == 1:
                final_pred = 2  # jaw_clench
            else:
                final_pred = 0  # idle

            dl_label = "move" if dl_pred == 1 else "other"
            svm_label = "jaw_clench" if svm_pred_raw == 1 else "idle"
            final_label = LABEL_MAP[final_pred]

            pred_history.append(final_pred)
            consensus_pred = 0
            for target_id in [1, 2]:
                if pred_history.count(target_id) >= PREDICTION_THRESHOLD:
                    consensus_pred = target_id
                    break

            print(
                f"[classifier_worker]  DL: {dl_label:<12} {dl_conf:.2f}"
                f"  |  SVM: {svm_label:<12} {svm_conf:.2f}"
                f"  →  {final_label}"
                f"  | Winner: {LABEL_MAP[consensus_pred]}"
            )

            if not shared_state.demo_override.value:
                shared_state.prediction.value = consensus_pred
                shared_state.pred_confidence.value = float(
                    dl_conf if consensus_pred == 1 else svm_conf
                )
            # else: demo keyboard has control — don't overwrite

            elapsed = time.perf_counter() - t_start
            remaining = CONFIGS["STRIDE_SEC"] - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                print(
                    f"[classifier_worker] ⚠ Loop took {elapsed:.3f}s, "
                    f"over stride budget of {CONFIGS['STRIDE_SEC']}s"
                )

    except Exception as e:
        print(f"[classifier_worker] Fatal error: {e}")

    finally:
        shared_state.classifier_running.value = False
        if stream is not None:
            stream.disconnect()
        print("[classifier_worker] Stream disconnected.")


# ── Offline test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from processing.preprocess_pipeline import (
        MNEPipeline,
        notch,
        bandpass,
        rereference,
        add_idle_class,
        epoch,
        extract_tensor,
    )

    print("Running offline inference test...")

    FIF_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data_collection",
            "annotated_fifs",
            "chengyi_4_21_1.fif",
        )
    )
    print(f"Loading: {FIF_PATH}")

    pipeline = MNEPipeline(FIF_PATH)
    pipeline.add(notch(60))
    pipeline.add(bandpass([(8, 13)]))
    pipeline.add(rereference())
    pipeline.add(add_idle_class(window_dur=3.0, idle_start_min=1.0))
    pipeline.add(epoch(tmin=0, tmax=3.0))
    raw = pipeline.run()

    LABEL_MAP_INV = {"idle": 0, "move": 1, "jaw_clench": 2}
    X, y_raw = extract_tensor(scale=True)(raw)
    auto_ids = raw._event_id
    remap = {v: LABEL_MAP_INV[k] for k, v in auto_ids.items() if k in LABEL_MAP_INV}
    y = np.array([remap[yi] for yi in y_raw])

    model = load_model()
    correct = 0
    results = []

    for i in range(len(X)):
        x = torch.tensor(X[i : i + 1], dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
            pred = output.argmax(1).item()
            conf = output[0, pred].item()
        correct += pred == y[i]
        results.append((LABEL_MAP[y[i]], LABEL_MAP[pred], conf))

    print(f"\nOffline accuracy: {correct / len(X):.3f} ({correct}/{len(X)})")
    print("\nSample predictions (true → pred, conf):")
    for true, pred, conf in results[:20]:
        match = "✓" if true == pred else "✗"
        print(f"  {match} {true:<12} → {pred:<12} {conf:.2f}")
