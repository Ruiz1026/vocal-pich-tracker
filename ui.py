import os
import re
from collections import deque

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except Exception:
    pg = None


class PianoKeyboardWidget(QtWidgets.QWidget):
    """Simple piano keyboard preview with active-key highlight."""

    WHITE_PCS = {0, 2, 4, 5, 7, 9, 11}
    BLACK_PCS = {1, 3, 6, 8, 10}

    def __init__(self, start_midi: int = 48, end_midi: int = 83):
        super().__init__()
        self.start_midi = start_midi
        self.end_midi = end_midi
        self.active_midi = None

        self.setMinimumHeight(128)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_active_midi(self, midi):
        if midi is None or midi < self.start_midi or midi > self.end_midi:
            midi = None
        if midi != self.active_midi:
            self.active_midi = midi
            self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        area = self.rect().adjusted(8, 8, -8, -8)
        if area.width() <= 4 or area.height() <= 4:
            return

        white_midis = [m for m in range(self.start_midi, self.end_midi + 1) if (m % 12) in self.WHITE_PCS]
        if not white_midis:
            return

        white_w = area.width() / len(white_midis)
        white_h = area.height()
        black_w = white_w * 0.62
        black_h = white_h * 0.62

        white_rects = {}

        x = float(area.left())
        white_border = QtGui.QPen(QtGui.QColor("#b8c9e3"))
        white_border.setWidth(1)

        for midi in white_midis:
            rect = QtCore.QRectF(x, float(area.top()), white_w, white_h)
            if midi == self.active_midi:
                fill = QtGui.QColor("#cde2ff")
            else:
                fill = QtGui.QColor("#ffffff")

            painter.fillRect(rect, fill)
            painter.setPen(white_border)
            painter.drawRect(rect)
            white_rects[midi] = rect

            # Mark C keys for orientation.
            if midi % 12 == 0:
                octave = midi // 12 - 1
                label_rect = QtCore.QRectF(rect.left(), rect.bottom() - 18, rect.width(), 16)
                painter.setPen(QtGui.QColor("#7a8dad"))
                painter.setFont(QtGui.QFont("Segoe UI", 8))
                painter.drawText(label_rect, QtCore.Qt.AlignCenter, f"C{octave}")

            x += white_w

        for midi in range(self.start_midi, self.end_midi + 1):
            if (midi % 12) not in self.BLACK_PCS:
                continue

            prev_white = midi - 1
            if prev_white not in white_rects:
                continue

            prev_rect = white_rects[prev_white]
            bx = prev_rect.right() - black_w / 2.0
            black_rect = QtCore.QRectF(bx, float(area.top()), black_w, black_h)

            if midi == self.active_midi:
                black_fill = QtGui.QColor("#2d7cf6")
                border = QtGui.QPen(QtGui.QColor("#1f5cc0"))
            else:
                black_fill = QtGui.QColor("#1f2633")
                border = QtGui.QPen(QtGui.QColor("#0f1623"))

            painter.fillRect(black_rect, black_fill)
            painter.setPen(border)
            painter.drawRect(black_rect)


class MainWindow(QtWidgets.QMainWindow):
    start_clicked = QtCore.pyqtSignal()
    stop_clicked = QtCore.pyqtSignal()
    mode_changed = QtCore.pyqtSignal(str)
    desktop_device_changed = QtCore.pyqtSignal(object)
    microphone_device_changed = QtCore.pyqtSignal(object)
    refresh_devices_clicked = QtCore.pyqtSignal()

    NOTE_TO_PC = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop/Microphone Audio Real-Time Pitch Detector")
        self.resize(980, 720)

        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "app_icon.svg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self._history = deque(maxlen=280)
        self._capture_mode = "desktop"
        self._is_running = False
        self._setup_style()

        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Vocal Pitch Tracker")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("Realtime pitch analysis from desktop audio or microphone input")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        panel_layout = QtWidgets.QGridLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setHorizontalSpacing(28)
        panel_layout.setVerticalSpacing(10)

        freq_caption = QtWidgets.QLabel("频率")
        freq_caption.setObjectName("caption")
        note_caption = QtWidgets.QLabel("音名")
        note_caption.setObjectName("caption")
        self.freq_value = QtWidgets.QLabel("-- Hz")
        self.freq_value.setObjectName("freqValue")
        self.note_value = QtWidgets.QLabel("--")
        self.note_value.setObjectName("noteValue")

        self.cents_value = QtWidgets.QLabel("Detune: -- cents")
        self.cents_value.setObjectName("subValue")
        self.conf_value = QtWidgets.QLabel("Confidence: --")
        self.conf_value.setObjectName("subValue")

        panel_layout.addWidget(freq_caption, 0, 0)
        panel_layout.addWidget(note_caption, 0, 1)
        panel_layout.addWidget(self.freq_value, 1, 0)
        panel_layout.addWidget(self.note_value, 1, 1)
        panel_layout.addWidget(self.cents_value, 2, 0)
        panel_layout.addWidget(self.conf_value, 2, 1)
        layout.addWidget(panel)

        source_panel = QtWidgets.QFrame()
        source_panel.setObjectName("panel")
        source_layout = QtWidgets.QGridLayout(source_panel)
        source_layout.setContentsMargins(16, 12, 16, 12)
        source_layout.setHorizontalSpacing(12)
        source_layout.setVerticalSpacing(10)

        mode_caption = QtWidgets.QLabel("采集模式")
        mode_caption.setObjectName("caption")
        source_layout.addWidget(mode_caption, 0, 0)

        mode_buttons = QtWidgets.QHBoxLayout()
        mode_buttons.setSpacing(8)
        self.desktop_mode_button = QtWidgets.QPushButton("桌面音频")
        self.desktop_mode_button.setObjectName("modeButton")
        self.desktop_mode_button.setCheckable(True)
        self.microphone_mode_button = QtWidgets.QPushButton("麦克风")
        self.microphone_mode_button.setObjectName("modeButton")
        self.microphone_mode_button.setCheckable(True)

        self.mode_button_group = QtWidgets.QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.mode_button_group.addButton(self.desktop_mode_button)
        self.mode_button_group.addButton(self.microphone_mode_button)
        self.desktop_mode_button.setChecked(True)

        mode_buttons.addWidget(self.desktop_mode_button)
        mode_buttons.addWidget(self.microphone_mode_button)
        mode_buttons.addStretch(1)
        source_layout.addLayout(mode_buttons, 0, 1, 1, 2)

        desktop_caption = QtWidgets.QLabel("桌面设备")
        desktop_caption.setObjectName("caption")
        self.desktop_device_combo = QtWidgets.QComboBox()
        self.desktop_device_combo.setObjectName("deviceCombo")
        source_layout.addWidget(desktop_caption, 1, 0)
        source_layout.addWidget(self.desktop_device_combo, 1, 1)

        self.refresh_devices_button = QtWidgets.QPushButton("刷新设备")
        self.refresh_devices_button.setObjectName("secondaryButton")
        source_layout.addWidget(self.refresh_devices_button, 1, 2)

        microphone_caption = QtWidgets.QLabel("麦克风设备")
        microphone_caption.setObjectName("caption")
        self.microphone_device_combo = QtWidgets.QComboBox()
        self.microphone_device_combo.setObjectName("deviceCombo")
        source_layout.addWidget(microphone_caption, 2, 0)
        source_layout.addWidget(self.microphone_device_combo, 2, 1, 1, 2)

        layout.addWidget(source_panel)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(10)
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self.desktop_mode_button.clicked.connect(self._on_desktop_mode_clicked)
        self.microphone_mode_button.clicked.connect(self._on_microphone_mode_clicked)
        self.desktop_device_combo.currentIndexChanged.connect(self._on_desktop_device_changed)
        self.microphone_device_combo.currentIndexChanged.connect(self._on_microphone_device_changed)
        self.refresh_devices_button.clicked.connect(lambda _checked=False: self.refresh_devices_clicked.emit())

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setObjectName("status")

        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        keyboard_panel = QtWidgets.QFrame()
        keyboard_panel.setObjectName("chartPanel")
        keyboard_layout = QtWidgets.QVBoxLayout(keyboard_panel)
        keyboard_layout.setContentsMargins(10, 10, 10, 10)
        keyboard_layout.setSpacing(8)

        keyboard_title = QtWidgets.QLabel("Piano Keyboard")
        keyboard_title.setObjectName("chartTitle")
        keyboard_layout.addWidget(keyboard_title)

        self.keyboard = PianoKeyboardWidget(start_midi=48, end_midi=83)
        keyboard_layout.addWidget(self.keyboard)
        layout.addWidget(keyboard_panel)

        chart_panel = QtWidgets.QFrame()
        chart_panel.setObjectName("chartPanel")
        chart_layout = QtWidgets.QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_layout.setSpacing(8)

        chart_title = QtWidgets.QLabel("Frequency Curve")
        chart_title.setObjectName("chartTitle")
        chart_layout.addWidget(chart_title)

        self.plot_widget = None
        self.curve = None
        if pg is not None:
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground("#f8fbff")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.28)
            self.plot_widget.setYRange(50, 1100)
            self.plot_widget.setLabel("left", "Frequency", "Hz")
            self.plot_widget.setLabel("bottom", "Time", "frames")
            self.plot_widget.getAxis("left").setTextPen(pg.mkColor("#4c6287"))
            self.plot_widget.getAxis("bottom").setTextPen(pg.mkColor("#4c6287"))
            self.plot_widget.getAxis("left").setPen(pg.mkPen("#c7d5ea"))
            self.plot_widget.getAxis("bottom").setPen(pg.mkPen("#c7d5ea"))
            self.curve = self.plot_widget.plot(pen=pg.mkPen(color="#2d7cf6", width=2.2))
            chart_layout.addWidget(self.plot_widget, stretch=1)
        else:
            fallback = QtWidgets.QLabel("Install pyqtgraph to show realtime frequency curve")
            fallback.setObjectName("fallback")
            chart_layout.addWidget(fallback)

        layout.addWidget(chart_panel, stretch=1)
        self._update_source_controls_enabled()

    def _setup_style(self) -> None:
        self.setStyleSheet(
            """
            #root {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 #f4f8ff, stop:1 #eef5ff);
            }
            QLabel {
                color: #21324f;
            }
            #title {
                font-size: 26px;
                font-weight: 700;
                color: #1a2b49;
            }
            #subtitle {
                font-size: 13px;
                color: #6079a2;
            }
            #panel, #chartPanel {
                background-color: #ffffff;
                border: 1px solid #d9e5f5;
                border-radius: 14px;
            }
            #caption {
                color: #6d84ab;
                font-size: 13px;
            }
            #freqValue, #noteValue {
                font-family: "Consolas", "Microsoft YaHei";
                font-size: 40px;
                font-weight: 700;
                color: #1e6ae6;
            }
            #subValue {
                font-size: 16px;
                color: #38537d;
            }
            #chartTitle {
                color: #4d6892;
                font-size: 13px;
                font-weight: 600;
                padding-left: 4px;
            }
            #primaryButton, #secondaryButton {
                min-height: 34px;
                min-width: 100px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                padding: 0 14px;
            }
            #primaryButton {
                background-color: #2d7cf6;
                color: #ffffff;
                border: none;
            }
            #primaryButton:hover {
                background-color: #4e93fb;
            }
            #primaryButton:disabled {
                background-color: #b8c8df;
                color: #eef3fb;
            }
            #secondaryButton {
                background-color: #ffffff;
                color: #365580;
                border: 1px solid #c9d8ee;
            }
            #secondaryButton:hover {
                background-color: #f1f6ff;
            }
            #secondaryButton:disabled {
                color: #96aac8;
                border-color: #d8e3f3;
            }
            #status {
                color: #4f688f;
                font-size: 13px;
            }
            #modeButton {
                min-height: 32px;
                min-width: 108px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                color: #33527e;
                background-color: #ffffff;
                border: 1px solid #c9d8ee;
                padding: 0 12px;
            }
            #modeButton:hover {
                background-color: #f1f6ff;
            }
            #modeButton:checked {
                background-color: #2d7cf6;
                color: #ffffff;
                border-color: #2d7cf6;
            }
            #modeButton:disabled {
                color: #96aac8;
                border-color: #d8e3f3;
            }
            #deviceCombo {
                min-height: 32px;
                border-radius: 8px;
                border: 1px solid #c9d8ee;
                background-color: #ffffff;
                color: #2b4368;
                padding: 0 10px;
            }
            #deviceCombo:disabled {
                color: #96aac8;
                border-color: #d8e3f3;
                background-color: #f7faff;
            }
            #fallback {
                color: #6a82aa;
                font-size: 13px;
            }
            """
        )

    @staticmethod
    def _note_name_to_midi(note_name: str):
        match = re.match(r"^([A-G])(#?)(-?\d+)$", note_name.strip())
        if not match:
            return None

        base = match.group(1)
        sharp = match.group(2)
        octave = int(match.group(3))
        key = base + sharp
        if key not in MainWindow.NOTE_TO_PC:
            return None

        return (octave + 1) * 12 + MainWindow.NOTE_TO_PC[key]

    def set_capture_mode(self, mode: str, emit_signal: bool = False) -> None:
        if mode not in {"desktop", "microphone"}:
            return
        self._capture_mode = mode
        self.desktop_mode_button.setChecked(mode == "desktop")
        self.microphone_mode_button.setChecked(mode == "microphone")
        self._update_source_controls_enabled()
        if emit_signal:
            self.mode_changed.emit(mode)

    def set_desktop_devices(self, devices, selected_id=None) -> None:
        self._set_device_combo_items(self.desktop_device_combo, devices, selected_id, "无可用桌面设备")

    def set_microphone_devices(self, devices, selected_id=None) -> None:
        self._set_device_combo_items(self.microphone_device_combo, devices, selected_id, "无可用麦克风设备")

    def set_running(self, running: bool) -> None:
        self._is_running = running
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self._update_source_controls_enabled()

    def _set_device_combo_items(self, combo: QtWidgets.QComboBox, devices, selected_id, empty_text: str) -> None:
        previous_id = combo.currentData()
        combo.blockSignals(True)
        combo.clear()

        for item in devices:
            combo.addItem(item["label"], item["id"])

        if combo.count() == 0:
            combo.addItem(empty_text, None)

        target_id = selected_id if selected_id is not None else previous_id
        matched = False
        if target_id is not None:
            for idx in range(combo.count()):
                if combo.itemData(idx) == target_id:
                    combo.setCurrentIndex(idx)
                    matched = True
                    break
        if not matched and combo.count() > 0:
            combo.setCurrentIndex(0)

        combo.blockSignals(False)
        self._update_source_controls_enabled()

    def _combo_has_valid_item(self, combo: QtWidgets.QComboBox) -> bool:
        return combo.count() > 0 and combo.itemData(combo.currentIndex()) is not None

    def _update_source_controls_enabled(self) -> None:
        editable = not self._is_running
        is_desktop = self._capture_mode == "desktop"
        is_microphone = self._capture_mode == "microphone"

        self.desktop_mode_button.setEnabled(editable)
        self.microphone_mode_button.setEnabled(editable)
        self.refresh_devices_button.setEnabled(editable)

        self.desktop_device_combo.setEnabled(editable and is_desktop and self._combo_has_valid_item(self.desktop_device_combo))
        self.microphone_device_combo.setEnabled(
            editable and is_microphone and self._combo_has_valid_item(self.microphone_device_combo)
        )

    def _on_desktop_mode_clicked(self, _checked=False) -> None:
        if self.desktop_mode_button.isChecked():
            self.set_capture_mode("desktop", emit_signal=True)

    def _on_microphone_mode_clicked(self, _checked=False) -> None:
        if self.microphone_mode_button.isChecked():
            self.set_capture_mode("microphone", emit_signal=True)

    def _on_desktop_device_changed(self, _index: int) -> None:
        self.desktop_device_changed.emit(self.desktop_device_combo.currentData())

    def _on_microphone_device_changed(self, _index: int) -> None:
        self.microphone_device_changed.emit(self.microphone_device_combo.currentData())

    def _on_start(self) -> None:
        self.set_running(True)
        self.start_clicked.emit()

    def _on_stop(self) -> None:
        self.set_running(False)
        self.stop_clicked.emit()

    @QtCore.pyqtSlot(str)
    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    @QtCore.pyqtSlot(object)
    def on_pitch_result(self, result) -> None:
        if result.frequency is None:
            self.freq_value.setText("-- Hz")
            self.note_value.setText("--")
            self.cents_value.setText("Detune: -- cents")
            self.conf_value.setText("Confidence: --")
            self._history.append(0.0)
            self.keyboard.set_active_midi(None)
        else:
            self.freq_value.setText(f"{result.frequency:7.2f} Hz")
            self.note_value.setText(result.note_name)
            self.cents_value.setText(f"Detune: {result.cents:+.1f} cents")
            self.conf_value.setText(f"Confidence: {result.confidence:.2f}")
            self._history.append(result.frequency)
            self.keyboard.set_active_midi(self._note_name_to_midi(result.note_name))

        if self.curve is not None and len(self._history) > 0:
            y = list(self._history)
            x = list(range(len(y)))
            self.curve.setData(x, y)

    def closeEvent(self, event):
        self.stop_clicked.emit()
        super().closeEvent(event)
