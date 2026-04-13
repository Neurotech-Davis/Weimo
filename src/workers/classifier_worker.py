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
EXCLUDE = {"Trigger", "Event"}
CONFIGS = {
    "N_CHANNELS": 8,
    "N_CLASSES": 3,
    "SFREQ": 300,
    "TRIAL_DUR": 3,
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

        all_ch = stream.info["ch_names"]
        eeg_picks = [ch for ch in all_ch if ch not in EXCLUDE]

        shared_state.classifier_running.value = True
        # LOOP
        while not shared_state.shutdown.is_set():
            pass

    except Exception as e:
        # TODO : add information in shared_memory to accept diagnostics
        print(f"[classifier_worker] Fatal error: {e}")

    finally:
        # TEARDOWN
        shared_state.classifier_running.value = False
        if stream is not None:
            stream.disconnect()
        print("[classifier_worker] Stream disconnected.")
