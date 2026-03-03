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
    window.mode_changed.connect(engine.set_capture_mode)
    window.desktop_device_changed.connect(engine.set_desktop_device)
    window.microphone_device_changed.connect(engine.set_microphone_device)

    engine.pitch_updated.connect(window.on_pitch_result)
    engine.status_changed.connect(window.set_status)

    def _refresh_device_lists(force_refresh: bool = True) -> None:
        desktop_devices = engine.list_desktop_devices(force_refresh=force_refresh)
        microphone_devices = engine.list_microphone_devices(force_refresh=force_refresh)
        window.set_desktop_devices(desktop_devices, engine.selected_desktop_device_id)
        window.set_microphone_devices(microphone_devices, engine.selected_microphone_device_id)
        engine.set_desktop_device(engine.selected_desktop_device_id)
        engine.set_microphone_device(engine.selected_microphone_device_id)

    window.refresh_devices_clicked.connect(lambda: _refresh_device_lists(True))

    def _show_error(msg: str) -> None:
        window.set_status(msg)
        QtWidgets.QMessageBox.critical(window, "Error", msg)
        window.set_running(False)

    engine.error_occurred.connect(_show_error)
    engine.set_capture_mode("desktop")
    window.set_capture_mode("desktop")
    _refresh_device_lists(force_refresh=True)

    window.show()
    code = app.exec_()

    # Ensure audio resources are released.
    engine.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
