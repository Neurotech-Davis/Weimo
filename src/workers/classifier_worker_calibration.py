"""
Calibration-buffer variant of classifier_worker.

Maintains a rolling buffer of recent EEG data (CALIB_SEC seconds).
Each inference step the entire buffer is filtered before extracting the
last N_TIMEPOINTS samples — closely replicating the whole-file preprocessing
used during training and eliminating filter edge-effect mismatch.

Requires ~CALIB_SEC seconds of warm-up before inference begins.
"""

import time
import sys
import os

import numpy as np
import torch
import mne

from mne_lsl.lsl import local_clock
from mne_lsl.stream import StreamLSL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml_model import models

MOVE_CONFIDENCE_THRESHOLD = 0.95
LABEL_MAP = {0: "other", 1: "move"}
EXCLUDE = {"Trigger", "Event"}

CONFIGS = {
    "N_CHANNELS": 8,
    "N_CLASSES": 2,
    "TRIAL_DUR": 3,
    "SFREQ": 300,
    "STRIDE_SEC": 1,
    "CALIB_SEC": 60,        # seconds of rolling buffer — must be >> TRIAL_DUR
}
CONFIGS["N_TIMEPOINTS"]   = int(CONFIGS["SFREQ"] * CONFIGS["TRIAL_DUR"]) + 1  # 901
CONFIGS["STRIDE_SAMPLES"] = int(CONFIGS["SFREQ"] * CONFIGS["STRIDE_SEC"])      # 300
CONFIGS["CALIB_SAMPLES"]  = int(CONFIGS["SFREQ"] * CONFIGS["CALIB_SEC"])       # 18000

MODEL_PATH = "../models/DeepConvNet_whole_file.pt"
STREAM_NAME = "WS-default"

CH_NAMES = [
    "EEG LE-Pz", "EEG F4-Pz", "EEG C4-Pz", "EEG P4-Pz",
    "EEG P3-Pz", "EEG C3-Pz", "EEG F3-Pz", "Pz",
]


# ── Preprocessing ──────────────────────────────────────────────────────────────


def preprocess_buffer(buffer: np.ndarray) -> torch.Tensor:
    """Filter the full rolling buffer, extract the last N_TIMEPOINTS samples.

    buffer shape: (n_channels, many_samples) — the entire rolling buffer
    Returns: (1, 1, 8, 901) float32 tensor
    """
    info = mne.create_info(ch_names=CH_NAMES, sfreq=CONFIGS["SFREQ"], ch_types="eeg")
    raw = mne.io.RawArray(buffer, info, verbose=False)

    raw.notch_filter(60.0, verbose=False)
    raw.filter(13, 30.0, verbose=False)
    raw.set_eeg_reference(ref_channels="average", verbose=False)

    X = raw.get_data()[:, -CONFIGS["N_TIMEPOINTS"]:]   # last 901 samples
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mean) / std

    return torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 901)


# ── Signal quality check ───────────────────────────────────────────────────────


REFERENCE_CHANNELS = {"Pz", "TRG"}


def check_signal_quality(data: np.ndarray, eeg_picks: list) -> bool:
    any_flat = False
    for i, ch in enumerate(eeg_picks):
        ch_std = data[i].std()
        ch_max = data[i].max()
        ch_min = data[i].min()
        flat = ch_std < 1e-6 and ch not in REFERENCE_CHANNELS
        if flat:
            any_flat = True
        flag = "⚠ FLAT" if flat else ""
        print(f"  {ch:<20} std={ch_std:.6f}  max={ch_max:.6f}  min={ch_min:.6f}  {flag}")
    return not any_flat


# ── LSL connection ─────────────────────────────────────────────────────────────


def attempt_LSL_connection(max_retries: int, retry_delay: int):
    stream = None
    for attempt in range(1, max_retries + 1):
        try:
            stream = StreamLSL(
                bufsize=CONFIGS["CALIB_SEC"] + 2.0,
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


# ── Model ──────────────────────────────────────────────────────────────────────


def load_model():
    model = models.DeepConvNet(CONFIGS["N_CHANNELS"], CONFIGS["N_CLASSES"]).to("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"[classifier_worker] Loaded model from {MODEL_PATH}")
    return model


# ── Worker ─────────────────────────────────────────────────────────────────────


def classifier_worker(shared_state):
    stream = None
    try:
        model = load_model()
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
        print(f"  Calibration buffer: {CONFIGS['CALIB_SEC']}s ({CONFIGS['CALIB_SAMPLES']} samples)")
        print("-------------------\n")

        shared_state.classifier_running.value = True

        # wait for first stride to be available
        time.sleep(CONFIGS["STRIDE_SEC"] + 0.5)

        rolling_buffer = np.zeros((len(eeg_picks), 0), dtype=np.float32)
        calibrated = False
        last_stride = None

        # LOOP
        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            # request only one stride of new data each iteration
            stride_data, timestamps = stream.get_data(
                winsize=CONFIGS["STRIDE_SEC"],
                picks=eeg_picks,
            )

            # freshness check
            stream_age = local_clock() - timestamps[-1]
            if stream_age > 0.5:
                print(f"[classifier_worker] ⚠ Stream is {stream_age:.2f}s stale, skipping")
                time.sleep(0.1)
                continue

            # identical stride check
            if last_stride is not None and np.array_equal(stride_data, last_stride):
                print("[classifier_worker] ⚠ Stride identical to last — possible stale data")
            last_stride = stride_data.copy()

            # append stride to rolling buffer, trim to CALIB_SAMPLES
            rolling_buffer = np.concatenate([rolling_buffer, stride_data], axis=1)
            if rolling_buffer.shape[1] > CONFIGS["CALIB_SAMPLES"]:
                rolling_buffer = rolling_buffer[:, -CONFIGS["CALIB_SAMPLES"]:]

            # calibration phase — accumulate until buffer is full
            if not calibrated:
                pct = rolling_buffer.shape[1] / CONFIGS["CALIB_SAMPLES"] * 100
                print(f"[classifier_worker] Calibrating... {pct:.0f}%  "
                      f"({rolling_buffer.shape[1]}/{CONFIGS['CALIB_SAMPLES']} samples)")
                if rolling_buffer.shape[1] >= CONFIGS["CALIB_SAMPLES"]:
                    calibrated = True
                    print("[classifier_worker] ✓ Calibration complete — starting inference")
                elapsed = time.perf_counter() - t_start
                time.sleep(max(0, CONFIGS["STRIDE_SEC"] - elapsed))
                continue

            # signal quality check on the latest stride
            print("[classifier_worker] Signal quality:")
            signal_ok = check_signal_quality(stride_data, eeg_picks)
            if not signal_ok:
                print("[classifier_worker] ⚠ Flatlined channel detected — skipping inference")
                time.sleep(0.1)
                continue

            # preprocess full buffer → extract last epoch
            x = preprocess_buffer(rolling_buffer)
            print(
                f"  [preprocess] raw    range: [{stride_data.min():.6f}, {stride_data.max():.6f}]  std: {stride_data.std():.6f}"
            )
            print(
                f"  [preprocess] scaled range: [{x.min():.3f}, {x.max():.3f}]  std: {x.std():.3f}"
            )
            if torch.isnan(x).any():
                print("[classifier_worker] ⚠ NaNs in tensor — skipping")
                continue

            # inference
            with torch.no_grad():
                output = model(x)
                pred = output.argmax(1).item()
                conf = output[0, pred].item()

            label = LABEL_MAP[pred]
            print(f"[classifier_worker] pred={label:<12} conf={conf:.2f}")

            if conf >= MOVE_CONFIDENCE_THRESHOLD:
                shared_state.prediction.value = pred
            else:
                shared_state.prediction.value = 0
            shared_state.pred_confidence.value = float(conf)

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
