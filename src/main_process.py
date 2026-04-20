### import
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

from core.shared_state import SharedState
from workers.eyetracker_worker import eyetracker_worker
from workers.classifier_worker import classifier_worker
from workers.envcam_worker import envcam_worker
from workers.lidar_worker import lidar_worker

# potentially can/should build out the PyQT window as its own module

import multiprocessing as mp


def main():
    shared_state = SharedState()  # all shared mem in one place

    p_tracker = mp.Process(target=eyetracker_worker, args=(shared_state,))
    p_classifier = mp.Process(target=classifier_worker, args=(shared_state,))
    p_envcam = mp.Process(target=envcam_worker, args=(shared_state,))
    # p_lidar = mp.Process(target=lidar_worker, args=(shared_state,))

    # process_arr = [p_tracker]
    process_arr = [p_tracker, p_classifier]
    # process_arr = [p_tracker, p_classifier, p_lidar]

    for proc in process_arr:
        proc.start()

    app = QApplication(sys.argv)
    window = MainWindow(shared_state)
    window.show()
    app.exec()

    shared_state.shutdown.set()
    for proc in process_arr:
        proc.join(timeout=3)


if __name__ == "__main__":
    mp.set_start_method("spawn")  # important for crossplatform safety
    main()
