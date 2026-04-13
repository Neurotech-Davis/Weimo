"""
minimal_lsl_test.py - validate DSI-7 LSL connection
run from src/: python workers/minimal_lsl_test.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
from datetime import datetime
import numpy as np
from mne_lsl.stream import StreamLSL

STREAM_NAME = "WS-default"
TRIAL_DUR = 3.0
EXCLUDE = {"Trigger", "Event"}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = f"lsl_test_log_{timestamp}.npz"


def attempt_LSL_connection(max_retries=10, retry_delay=2):
    stream = None
    for attempt in range(1, max_retries + 1):
        try:
            stream = StreamLSL(bufsize=TRIAL_DUR + 1.0, name=STREAM_NAME).connect()
            print(f"Connected on attempt {attempt}.")
            break
        except Exception as e:
            print(
                f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
    return stream


def main():
    stream = attempt_LSL_connection()
    if stream is None:
        raise RuntimeError("Could not connect. Is dsi2lsl running?")

    all_ch = stream.info["ch_names"]
    sfreq = stream.info["sfreq"]
    eeg_picks = [ch for ch in all_ch if ch not in EXCLUDE]

    # print header info — first thing to validate
    print(f"\n--- Stream Info ---")
    print(f"  All Channels ({len(all_ch)}): {all_ch}")
    print(f"  Selected Channels ({len(eeg_picks)}): {eeg_picks}")
    print(f"  Sfreq: {sfreq} Hz")
    print(f"  Expected shape per window: ({len(eeg_picks)}, {int(sfreq * TRIAL_DUR)})")
    print(f"-------------------\n")

    time.sleep(TRIAL_DUR + 0.5)  # let buffer fill

    chunks = []
    print("Collecting 10 windows. Ctrl+C to stop early...\n")
    try:
        for i in range(10):
            data, timestamps = stream.get_data(winsize=TRIAL_DUR, picks=eeg_picks)
            chunks.append(data)

            # live sanity checks — the important stuff
            print(
                f"[{i + 1:02d}] shape={data.shape} | "
                f"mean={data.mean():.4f} | "
                f"std={data.std():.4f}  | "
                f"min={data.min():.4f} | "
                f"max={data.max():.4f}"
            )

            # red flags to watch for:
            if data.std() < 1e-6:
                print(
                    "  ⚠ WARNING: near-zero variance — flatline or disconnected electrode?"
                )
            if np.isnan(data).any():
                print("  ⚠ WARNING: NaNs in stream")
            if np.abs(data).max() > 500:
                print("  ⚠ WARNING: large amplitude spike — check electrode contact")

            time.sleep(0.25)

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        stream.disconnect()
        print("Stream disconnected.")

    # save to file for offline analysis — your instinct
    if chunks:
        X = np.stack(chunks, axis=0)  # (n_windows, n_channels, n_timepoints)
        np.savez(LOG_PATH, X=X, ch_names=eeg_picks, sfreq=sfreq)
        print(f"\nSaved {X.shape} → {LOG_PATH}")
        print("Load with: data = np.load('lsl_test_log.npz')")


if __name__ == "__main__":
    main()
