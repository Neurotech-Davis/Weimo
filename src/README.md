## src

/src is where we defne the source of truth for the demo. Functions written here have to be conscious of the face that they will be used as concurrent workers by python's `multiprocessing` module. Read contract below for how to adhere to worker semantics.

## Installation

1. Create and activate venv. Guide on this in project root.
   - Needed in order for `opencv-python-headless` to not conflict with `opencv-python`, which bundles conflicting UI packages.
2. `pip install -r requirements.txt`
   - If on linux, errors might be encountered in running the PyQt6 process. If so, install the missing libraries: `sudo apt install libxcb-cursor0`
3. From `/src`: `python3 main_process.py`

---

When using a new computer, adjustments need to be made to:

- Default camera index
- Camera resolution, the shared_memory frame buffer needs to be resized to accomodate.

## Usage

1. Launch the DSI-Streamer GUI and do initial signal checking. Use
   - Guide to debugging common signal quality pitfalls - [DSI Signal Quality YT Video](https://www.youtube.com/watch?v=6juYPfUCEbA)
     - [Link to notes](https://vault.tynerlab.com/College/Clubs/Weimo-DSI-Streamer#signal)
2. Iniate `dsi2lslGUI`, the LSL data streamer for the DSI-7 headset. _Note_: DSI-Streamer **HAS** to be closed in order to free the COM port for the LSL streamer GUI.
   - [Link to Wearable Sensing Docs](https://support.wearablesensing.com/examples/lsl/gui.html)
   - [Link to dsi2lsl download](https://github.com/labstreaminglayer/App-WearableSensing) - There is a releases page to the left with a zip file with downloads.
3. After the dsi2lslGUI is initiated, run the script. From `/src`: `python3 main_process.py`

## Worker contracts

Workers need to be aware that they are run continously, can be thought of as having a "god-looop". Outputs will not be read unless they write to the `shared_state` parameter, in accordance with the values defined in `/src/core/shared_state.py`.

- Shared state should be modified for inter-process-communication.

### Guide for defining workers:

Every worker must follow the following structure:

```python
def your_worker(shared_state):
    # 1. SETUP — runs once, initialise hardware/models/connections

    # 2. LOOP — runs until shutdown signal
    while not shared_state.shutdown.is_set():
        # read from hardware or shared_state
        # process
        # write results to shared_state

    # 3. TEARDOWN — release hardware, close connections
```

### Adapting Existing Code

Existing code doesn't need to be rewritten, instead code can be adapted and called by the worker wrapper:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "your_module"))

from your_existing_script import setup, process

def your_worker(shared_state):
    device = setup()
    while not shared_state.shutdown.is_set():
        result = process(device)
        shared_state.your_field.value = result
    device.close()
```

## Resources

MNE guide to LSL streams: [link](https://mne.tools/mne-lsl/stable/generated/tutorials/00_introduction.html)
