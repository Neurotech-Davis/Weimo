"""
For this worker to work, it has to be connected to the DSI-7s LSL streamer
- (initialize the streaming layer first)
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
LABEL_MAP = {0: "idle", 1: "move", 2: "jaw_clench"}
EXCLUDE = {"Trigger", "Event"}

CONFIGS = {
    "N_CHANNELS": 8,
    "N_CLASSES": 3,
    "TRIAL_DUR": 3,
    "SFREQ": 300,
    "STRIDE_SEC": 1,
}
CONFIGS["N_TIMEPOINTS"] = int(CONFIGS["SFREQ"] * CONFIGS["TRIAL_DUR"])

MODEL_PATH = "./models/DeepConvNet2.pt"
STREAM_NAME = "WS-default"


# ── Preprocessing ─────────────────────────────────────────────────────────────


def preprocess_epoch(data: np.ndarray) -> torch.Tensor:
    """
    Replicates the offline training pipeline on a single (n_channels, n_timepoints) window.
    Channel order must match training:
      ['EEG LE-Pz', 'EEG F4-Pz', 'EEG C4-Pz', 'EEG P4-Pz',
       'EEG P3-Pz', 'EEG C3-Pz', 'EEG F3-Pz', 'Pz']
    Returns: (1, 1, 8, 900) float32 tensor
    """
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
    info = mne.create_info(ch_names=ch_names, sfreq=CONFIGS["SFREQ"], ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    raw.notch_filter(60.0, method="iir", verbose=False)
    raw.filter(8.0, 30.0, method="iir", verbose=False)
    raw.set_eeg_reference(ref_channels="average", verbose=False)

    X = raw.get_data()[:, : CONFIGS["N_TIMEPOINTS"]]  # clip to exactly 900
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mean) / std

    return torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 8, 900)


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
        flag = "⚠ FLAT" if flat else ""
        print(
            f"  {ch:<20} std={ch_std:.6f}  max={ch_max:.6f}  min={ch_min:.6f}  {flag}"
        )
    return not any_flat


# ── LSL connection ────────────────────────────────────────────────────────────


def attempt_LSL_connection(max_retries: int, retry_delay: int):
    stream = None
    for attempt in range(1, max_retries + 1):
        try:
            stream = StreamLSL(
                bufsize=CONFIGS["TRIAL_DUR"] + 1.0, name=STREAM_NAME
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
    print(f"[classifier_worker] Loaded model from {MODEL_PATH}")
    return model


# ── Worker ────────────────────────────────────────────────────────────────────


def classifier_worker(shared_state):
    stream = None
    try:
        # SETUP
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
        print(
            f"  Expected shape: ({len(eeg_picks)}, {int(sfreq * CONFIGS['TRIAL_DUR'])})"
        )
        print("-------------------\n")

        shared_state.classifier_running.value = True

        # let buffer fill before first read
        time.sleep(CONFIGS["TRIAL_DUR"] + 0.5)

        last_data = None

        # LOOP
        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            data, timestamps = stream.get_data(
                winsize=CONFIGS["TRIAL_DUR"],
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
            print("[classifier_worker] Signal quality:")
            signal_ok = check_signal_quality(data, eeg_picks)
            if not signal_ok:
                print(
                    "[classifier_worker] ⚠ Flatlined channel detected — skipping inference"
                )
                time.sleep(0.1)
                continue

            # preprocess
            x = preprocess_epoch(data)
            # DEBUG — remove once hardware validated
            print(
                f"  [preprocess] raw    range: [{data.min():.6f}, {data.max():.6f}]  std: {data.std():.6f}"
            )
            print(
                f"  [preprocess] scaled range: [{x.min():.3f}, {x.max():.3f}]  std: {x.std():.3f}"
            )
            if torch.isnan(x).any():
                print("[classifier_worker] ⚠ NaNs in tensor — skipping")
                continue

            # inference — model outputs probabilities directly (softmax in forward())
            with torch.no_grad():
                output = model(x)
                pred = output.argmax(1).item()
                conf = output[0, pred].item()

            label = LABEL_MAP[pred]
            print(f"[classifier_worker] pred={label:<12} conf={conf:.2f}")

            # only write confident predictions — motor_worker reads this
            if conf >= MOVE_CONFIDENCE_THRESHOLD:
                shared_state.prediction.value = pred
            else:
                shared_state.prediction.value = 0  # treat low confidence as idle
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
            "chengyi_4_8_1.fif",
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
