"""
Connects to the DSI-7 LSL stream and records all EEG data to a .fif file.
Run alongside classifier_worker — both read from the same stream independently.

Usage: launched automatically by main_process.py when --record-eeg is passed.
Output: data_collection/recorded_eeg/<timestamp>.fif
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

from mne_lsl.lsl import local_clock
from mne_lsl.stream import StreamLSL
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

STREAM_NAME = "WS-default"
EXCLUDE     = {"Trigger", "Event"}
CHUNK_SEC   = 1.0   # pull data in 1-second chunks
OUTPUT_DIR  = os.path.join(
    os.path.dirname(__file__), "..", "..", "data_collection", "recorded_eeg"
)


def eeg_recorder_worker(shared_state, output_path: str = None):
    stream = None
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if output_path is None:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"{timestamp}.fif")

        print(f"[eeg_recorder] Connecting to LSL stream '{STREAM_NAME}'...")
        for attempt in range(1, 11):
            try:
                stream = StreamLSL(bufsize=CHUNK_SEC + 1.0, name=STREAM_NAME).connect()
                print(f"[eeg_recorder] Connected on attempt {attempt}.")
                break
            except Exception as e:
                print(f"[eeg_recorder] Attempt {attempt}/10 failed: {e}. Retrying in 2s...")
                time.sleep(2)

        if stream is None:
            raise RuntimeError("[eeg_recorder] Could not connect to LSL stream.")

        all_ch  = stream.info["ch_names"]
        sfreq   = stream.info["sfreq"]
        eeg_ch  = [ch for ch in all_ch if ch not in EXCLUDE]

        print(f"[eeg_recorder] Recording {len(eeg_ch)} channels at {sfreq} Hz → {output_path}")

        info = mne.create_info(ch_names=eeg_ch, sfreq=sfreq, ch_types="eeg")

        chunks  = []
        time.sleep(CHUNK_SEC)  # let buffer fill

        while not shared_state.shutdown.is_set():
            t_start = time.perf_counter()

            data, timestamps = stream.get_data(winsize=CHUNK_SEC, picks=eeg_ch)
            # data: (n_channels, n_samples)

            # freshness check — skip stale chunks
            age = local_clock() - timestamps[-1]
            if age < 2.0:
                chunks.append(data.copy())

            elapsed   = time.perf_counter() - t_start
            remaining = CHUNK_SEC - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except Exception as e:
        print(f"[eeg_recorder] Fatal error: {e}")

    finally:
        if stream is not None:
            stream.disconnect()

        if chunks:
            all_data = np.concatenate(chunks, axis=1)  # (n_channels, total_samples)
            raw = mne.io.RawArray(all_data, info, verbose=False)
            raw.save(output_path, overwrite=True)
            duration = all_data.shape[1] / sfreq
            print(f"[eeg_recorder] Saved {duration:.1f}s of EEG to {output_path}")
        else:
            print("[eeg_recorder] No data recorded.")

        print("[eeg_recorder] Done.")
