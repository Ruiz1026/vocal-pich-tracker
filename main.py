import os
import sys

from PyQt5 import QtGui, QtWidgets

from audio_capture import AudioEngine
from ui import MainWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "app_icon.svg")
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))

    window = MainWindow()
    # Balanced defaults: keep fast response while preserving vocal-pitch robustness.
    engine = AudioEngine(update_interval_ms=55, frame_size=4096, allow_microphone_fallback=False)

    window.start_clicked.connect(engine.start)
    window.stop_clicked.connect(engine.stop)

    engine.pitch_updated.connect(window.on_pitch_result)
    engine.status_changed.connect(window.set_status)

    def _show_error(msg: str) -> None:
        window.set_status(msg)
        QtWidgets.QMessageBox.critical(window, "Error", msg)
        window.start_button.setEnabled(True)
        window.stop_button.setEnabled(False)

    engine.error_occurred.connect(_show_error)

    window.show()
    code = app.exec_()

    # Ensure audio resources are released.
    engine.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
