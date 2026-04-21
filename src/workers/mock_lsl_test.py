"""
This is a mock for the streamer to test performance
"""

import os
import time
import mne
from mne_lsl.player import PlayerLSL

# Define paths relative to src/workers
# DATA_DIR = "../../data_collection/raw_fif"
DATA_DIR = "../../data_collection/annotated_fifs"
FILE_NAME = "chengyi_4_8_0.fif"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)


def start_mock():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Could not find {FILE_PATH}")
        return

    print(f"Loading {FILE_NAME} for streaming...")

    # name="WS-default" matches the STREAM_NAME in your classifier_worker
    # chunk_size=10 ensures a steady flow of samples (30Hz updates at 300Hz sfreq)

    raw = mne.io.read_raw_fif(FILE_PATH, preload=True)
    midpoint = raw.times[-1] / 2
    raw.crop(tmin=midpoint)
    player = PlayerLSL(raw, chunk_size=10, name="WS-default")
    # player = PlayerLSL(FILE_PATH, chunk_size=10, name="WS-default")
    player.start()

    player.start()
    print("LSL Stream 'WS-default' is LIVE. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        player.stop()
        print("\nStream stopped.")


if __name__ == "__main__":
    start_mock()
