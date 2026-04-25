### import
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

from core.shared_state import SharedState
from workers.eyetracker_worker import eyetracker_worker
from workers.classifier_worker import classifier_worker
from workers.eeg_recorder_worker import eeg_recorder_worker
from workers.motor_worker import motor_worker
from workers.pathcam_worker import pathcam_worker
from workers.pathfinding_worker import pathfinding_worker

import multiprocessing as mp


def main():
    record_eeg = "--record-eeg" in sys.argv
    mock_classifier = "--mock-classifier" in sys.argv
    mixed_classifier = "--mixed-classifier" in sys.argv
    demo_classifier = "--demo-classifier" in sys.argv
    shared_state = SharedState()  # all shared mem in one place
    shared_state.mock_classifier.value = mock_classifier
    shared_state.demo_classifier.value = demo_classifier

    # Daemon=true means they are killed when parent dies. This is desired UNLESS they have important cleanup in their "finally" block, like the motor worker which wants to shut off the car.
    p_tracker = mp.Process(target=eyetracker_worker, args=(shared_state,), daemon=True)
    p_pathcam = mp.Process(target=pathcam_worker, args=(shared_state,), daemon=True)
    p_classifier = mp.Process(
        target=classifier_worker, args=(shared_state,), daemon=True
    )
    p_recorder = mp.Process(target=eeg_recorder_worker, args=(shared_state,))
    p_motor = mp.Process(target=motor_worker, args=(shared_state,))
    p_pathfinding = mp.Process(
        target=pathfinding_worker, args=(shared_state,), daemon=True
    )

    # process_arr = [p_tracker]
    # process_arr = [p_tracker, p_pathfinding]
    process_arr = [p_tracker, p_pathfinding, p_motor, p_pathcam]
    if record_eeg:
        process_arr.append(p_recorder)

    shared_state.mock_classifier.value = mock_classifier
    if not mock_classifier:
        process_arr.append(p_classifier)

    for proc in process_arr:
        proc.start()

    app = QApplication(sys.argv)
    window = MainWindow(
        shared_state, mock_classifier=mock_classifier, mixed_classifier=mixed_classifier,
        demo_classifier=demo_classifier
    )
    window.show()
    app.exec()

    shared_state.shutdown.set()
    for proc in process_arr:
        proc.join(timeout=3)


if __name__ == "__main__":
    mp.set_start_method("spawn")  # important for crossplatform safety
    main()