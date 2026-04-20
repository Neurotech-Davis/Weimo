"""
For this worker to work, it has to be connected to the DSI-7s LSL streamer
- (initalize the streaming layer first)
"""

import time
import sys
import os

import numpy as np
import torch

from mne_lsl.stream import StreamLSL
from mne.time_frequency import psd_array_welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml_model import models
from processing import preprocess_pipeline

LABEL_MAP = {0: "idle", 1: "move", 2: "jaw_clench"}
EXCLUDE = {"TRG"}  # excluding the trigger
CONFIGS = {
    "N_CHANNELS": 8,
    "N_CLASSES": 3,
    "TRIAL_DUR": 3,
    "SFREQ": 300,
    "STRIDE_SEC": 0.25,
}
CONFIGS["N_TIMEPOINTS"] = int(CONFIGS["SFREQ"] * CONFIGS["TRIAL_DUR"])
MODEL_PATH = "../models/best_model.pt"  # name is arbitrary
STREAM_NAME = "WS-default"


def preprocess_epoch(data: np.ndarray):  # -> torch.Tensor:
    # use preprocess_pipeline import for functions
    return data


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
                f"[classifier_worker] Attempt {attempt}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
    return stream


def load_model():
    # loading to CPU, investigate whether this is optimal (likely is, super small scale inference)
    model = models.DeepConvNet(CONFIGS["N_CHANNELS"], CONFIGS["N_CLASSES"]).to("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"[classifier_worker] Loaded model from {MODEL_PATH}")
    return model


def classifier_worker(shared_state):
    stream = None
    try:
        # SETUP
        model = load_model()

        print(f"[classifier_worker] Connecting to LSL stream '{STREAM_NAME}'...")
        stream = attempt_LSL_connection(max_retries=10, retry_delay=2)
        if stream is None:
            raise RuntimeError(
                f"[classifier_worker] Could not connect to LSL stream '{STREAM_NAME}'. Is dsi2lsl running?"
            )

        all_ch = stream.info[
            "ch_names"
        ]  # DSI-7: ['Pz', 'F4', 'C4', 'P4', 'P3', 'C3', 'F3', 'TRG']
        sfreq = stream.info["sfreq"]  # DSI-7 : 300
        eeg_picks = [ch for ch in all_ch if ch not in EXCLUDE]  # removing 'TRG'

        # print header info — first thing to validate
        print("\n--- Stream Info ---")
        print(f"  All Channels ({len(all_ch)}): {all_ch}")
        print(f"  Selected Channels ({len(eeg_picks)}): {eeg_picks}")
        print(f"  Sfreq: {sfreq} Hz")
        print(
            f"  Expected shape per window: ({len(eeg_picks)}, {int(sfreq * CONFIGS['TRIAL_DUR'])})"
        )
        print("-------------------\n")

        shared_state.classifier_running.value = True

        # LOOP
        time.sleep(CONFIGS["TRIAL_DUR"] + 0.5)  # let buffer fill before first read
        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            # get_data always returns the MOST RECENT winsize seconds from the buffer
            data, timestamps = stream.get_data(
                winsize=CONFIGS["TRIAL_DUR"],
                picks=eeg_picks,
            )
            # data shape: (n_channels, n_timepoints) e.g. (7, 900)

            # freshness check — is the newest sample actually recent?
            stream_age = time.time() - timestamps[-1]
            if stream_age > 0.5:
                print(
                    f"[classifier_worker] ⚠ Stream is {stream_age:.2f}s stale, skipping window"
                )
                time.sleep(0.1)
                continue

            # preprocess + inference
            x = preprocess_epoch(data)  # (1, 1, n_channels, n_timepoints)
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(1).item()
                conf = torch.softmax(logits, dim=1)[0, pred].item()

            label = LABEL_MAP[pred]
            print(f"[classifier_worker] pred={label:<12} conf={conf:.2f}")

            shared_state.prediction.value = pred
            elapsed = time.perf_counter() - t_start
            remaining = CONFIGS["STRIDE_SEC"] - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                print(
                    f"[classifier_worker] ⚠ Loop took {elapsed:.3f}s, over stride budget of {CONFIGS['STRIDE_SEC']}s"
                )

    except Exception as e:
        # TODO : add information in shared_memory to accept diagnostics
        print(f"[classifier_worker] Fatal error: {e}")

    finally:
        # TEARDOWN
        shared_state.classifier_running.value = False
        if stream is not None:
            stream.disconnect()
        print("[classifier_worker] Stream disconnected.")
