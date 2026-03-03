import platform
import queue
import threading
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
from PyQt5 import QtCore

from pitch_detector import PitchDetector

try:
    import soundcard as sc
except Exception:
    sc = None


@dataclass
class PitchResult:
    frequency: Optional[float]
    note_name: str
    cents: float
    confidence: float


class AudioEngine(QtCore.QObject):
    """Capture desktop audio and publish real-time pitch detection results."""

    pitch_updated = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        update_interval_ms: int = 60,
        frame_size: int = 4096,
        allow_microphone_fallback: bool = False,
    ):
        super().__init__()
        self.update_interval_ms = update_interval_ms
        self.frame_size = frame_size
        self.allow_microphone_fallback = allow_microphone_fallback

        self.detector = PitchDetector(
            min_freq=75.0,
            max_freq=1250.0,
            yin_threshold=0.18,
            min_rms=2e-5,
            max_cmnd_without_threshold_hit=0.5,
            min_confidence=0.22,
        )
        # 0..1, larger means stronger center-vocal preference in stereo content.
        self.vocal_focus_strength = 0.8

        self.stream = None
        self.sample_rate = 48000
        self.channels = 2
        self._running = False

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._worker_thread: Optional[threading.Thread] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_stop = threading.Event()
        self._sc_recorder = None
        self._sc_candidates = []
        self._sc_candidate_idx = -1
        self._sc_silence_frames = 0
        self._window = np.hanning(self.frame_size).astype(np.float32)
        self._secondary_frame_size = 2048
        self._window_secondary = np.hanning(self._secondary_frame_size).astype(np.float32)
        self._probe_info = ""
        self.capture_mode = "desktop"
        self.selected_microphone_device_id: Optional[str] = None
        self.selected_desktop_device_id: Optional[str] = None
        self._microphone_sources: List[dict] = []
        self._desktop_sources: List[dict] = []

        # Temporal stabilizer for smoother real-time tracking.
        self._freq_history = deque(maxlen=5)
        self._last_output_freq: Optional[float] = None
        self._last_confidence: float = 0.0
        self._missed_updates = 0
        self._hold_frames = max(1, int(280 / max(1, self.update_interval_ms)))
        # Guard against accompaniment-driven pitch jumps in dense CD mixes.
        self._max_jump_semitones = 7.5
        self._jump_confirm_frames = 2
        self._high_jump_conf_bypass = 0.52
        self._high_note_threshold_hz = 349.23  # F4
        self._pending_jump_freq: Optional[float] = None
        self._pending_jump_count = 0
        self._strict_min_confidence = 0.26
        self._consensus_max_semitone_diff = 1.0

    def set_capture_mode(self, mode: str) -> None:
        if mode not in {"desktop", "microphone"}:
            return
        self.capture_mode = mode

    def set_microphone_device(self, source_id: Optional[str]) -> None:
        self.selected_microphone_device_id = source_id or None

    def set_desktop_device(self, source_id: Optional[str]) -> None:
        self.selected_desktop_device_id = source_id or None

    def list_microphone_devices(self, force_refresh: bool = False) -> List[dict]:
        if force_refresh or not self._microphone_sources:
            self._microphone_sources = self._enumerate_microphone_sources()

        if not self._microphone_sources:
            self.selected_microphone_device_id = None
            return []

        valid_ids = {item["id"] for item in self._microphone_sources}
        if self.selected_microphone_device_id not in valid_ids:
            default_item = next((item for item in self._microphone_sources if item.get("is_default")), None)
            self.selected_microphone_device_id = (default_item or self._microphone_sources[0])["id"]

        return [{"id": item["id"], "label": item["label"]} for item in self._microphone_sources]

    def list_desktop_devices(self, force_refresh: bool = False) -> List[dict]:
        if force_refresh or not self._desktop_sources:
            self._desktop_sources = self._enumerate_desktop_sources()

        if not self._desktop_sources:
            self.selected_desktop_device_id = None
            return []

        valid_ids = {item["id"] for item in self._desktop_sources}
        if self.selected_desktop_device_id not in valid_ids:
            default_item = next((item for item in self._desktop_sources if item.get("is_default")), None)
            self.selected_desktop_device_id = (default_item or self._desktop_sources[0])["id"]

        return [{"id": item["id"], "label": item["label"]} for item in self._desktop_sources]

    @staticmethod
    def _safe_default_device(index: int) -> Optional[int]:
        try:
            pair = sd.default.device
            if pair is None:
                return None
            value = pair[index]
            if value is None:
                return None
            value = int(value)
            return value if value >= 0 else None
        except Exception:
            return None

    @staticmethod
    def _is_loopback_name(name_lower: str) -> bool:
        return any(k in name_lower for k in ("loopback", "loop back", "回送", "环回", "回环", "回采"))

    @staticmethod
    def _is_stereo_mix_name(name_lower: str) -> bool:
        return any(k in name_lower for k in ("stereo mix", "stereomix", "what u hear", "wave out", "mixed output", "立体声混音", "混音"))

    @staticmethod
    def _hostapi_name(hostapis: List[dict], hostapi_idx: int) -> str:
        if hostapi_idx < 0 or hostapi_idx >= len(hostapis):
            return "Unknown API"
        return str(hostapis[hostapi_idx].get("name", "Unknown API"))

    @staticmethod
    def _find_source_by_id(source_id: Optional[str], items: List[dict]) -> Optional[dict]:
        if not source_id:
            return None
        for item in items:
            if item.get("id") == source_id:
                return item
        return None

    def _enumerate_microphone_sources(self) -> List[dict]:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_input = self._safe_default_device(0)

        sources: List[dict] = []
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue

            name = str(dev.get("name", ""))
            name_lower = name.lower()
            if self._is_loopback_name(name_lower) or self._is_stereo_mix_name(name_lower):
                continue

            hostapi_idx = int(dev.get("hostapi", -1))
            hostapi_name = self._hostapi_name(hostapis, hostapi_idx)
            is_default = idx == default_input
            default_tag = " [默认]" if is_default else ""
            sources.append(
                {
                    "id": f"mic:{idx}",
                    "kind": "microphone_input",
                    "device_index": idx,
                    "is_default": is_default,
                    "label": f"{name} ({hostapi_name}){default_tag}",
                }
            )

        sources.sort(key=lambda item: (not item.get("is_default", False), item["label"].lower()))
        return sources

    @staticmethod
    def _find_wasapi_hostapi_index(hostapis: List[dict]) -> Optional[int]:
        for idx, host in enumerate(hostapis):
            if "WASAPI" in str(host.get("name", "")).upper():
                return idx
        return None

    def _enumerate_desktop_sources(self) -> List[dict]:
        if platform.system() != "Windows":
            return []

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        wasapi_index = self._find_wasapi_hostapi_index(hostapis)
        if wasapi_index is None:
            return []

        default_output = self._safe_default_device(1)
        sources: List[dict] = []
        seen_ids = set()

        for idx, dev in enumerate(devices):
            if int(dev.get("hostapi", -1)) != wasapi_index:
                continue
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", ""))
            is_default = idx == default_output
            default_tag = " [默认]" if is_default else ""
            source_id = f"desk_out:{idx}"
            seen_ids.add(source_id)
            sources.append(
                {
                    "id": source_id,
                    "kind": "wasapi_output_loopback",
                    "device_index": idx,
                    "is_default": is_default,
                    "hostapi_index": wasapi_index,
                    "label": f"{name} (WASAPI 输出回采){default_tag}",
                }
            )

        for idx, dev in enumerate(devices):
            if int(dev.get("hostapi", -1)) != wasapi_index:
                continue
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", ""))
            if not self._is_loopback_name(name.lower()):
                continue
            source_id = f"desk_loop_in:{idx}"
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            sources.append(
                {
                    "id": source_id,
                    "kind": "wasapi_loopback_input",
                    "device_index": idx,
                    "is_default": False,
                    "hostapi_index": wasapi_index,
                    "label": f"{name} (WASAPI 环回输入)",
                }
            )

        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", ""))
            if not self._is_stereo_mix_name(name.lower()):
                continue
            source_id = f"desk_mix:{idx}"
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            hostapi_name = self._hostapi_name(hostapis, int(dev.get("hostapi", -1)))
            sources.append(
                {
                    "id": source_id,
                    "kind": "stereo_mix_input",
                    "device_index": idx,
                    "is_default": False,
                    "hostapi_index": int(dev.get("hostapi", -1)),
                    "label": f"{name} ({hostapi_name}, Stereo Mix)",
                }
            )

        sources.sort(key=lambda item: (not item.get("is_default", False), item["label"].lower()))
        return sources

    def start(self) -> None:
        if self._running:
            return

        try:
            stream_configs = self._build_stream_configs()
            errors: List[str] = []
            started_mode = None

            for mode_name, stream_kwargs in stream_configs:
                try:
                    self.stream = sd.InputStream(callback=self._audio_callback, **stream_kwargs)
                    self.stream.start()
                    self.sample_rate = int(stream_kwargs["samplerate"])
                    self.channels = int(stream_kwargs["channels"])
                    started_mode = mode_name
                    break
                except Exception as exc:
                    errors.append(f"{mode_name}: {exc}")
                    try:
                        if self.stream is not None:
                            self.stream.close()
                    except Exception:
                        pass
                    self.stream = None

            desktop_source = self._find_source_by_id(self.selected_desktop_device_id, self._desktop_sources)
            should_try_soundcard_fallback = (
                self.capture_mode == "desktop"
                and platform.system() == "Windows"
                and (desktop_source is None or desktop_source.get("kind") == "wasapi_output_loopback")
            )
            if started_mode is None and should_try_soundcard_fallback:
                try:
                    self._start_soundcard_loopback()
                    started_mode = "soundcard default-speaker loopback"
                except Exception as exc:
                    errors.append(f"soundcard default-speaker loopback: {exc}")
                    try:
                        if sc is None:
                            errors.append("soundcard probe: package not installed")
                        else:
                            names = [str(getattr(m, "name", "")) for m in sc.all_microphones(include_loopback=True)]
                            if names:
                                errors.append("soundcard probe microphones: " + "; ".join(names[:6]))
                            else:
                                errors.append("soundcard probe microphones: none")
                    except TypeError:
                        try:
                            names = [str(getattr(m, "name", "")) for m in sc.all_microphones()]
                            if names:
                                errors.append("soundcard probe microphones(no include_loopback): " + "; ".join(names[:6]))
                            else:
                                errors.append("soundcard probe microphones(no include_loopback): none")
                        except Exception:
                            pass
                    except Exception:
                        pass

            if started_mode is None:
                joined = "\n".join(errors) if errors else "No capture strategy available"
                msg = f"All capture strategies failed:\n{joined}"
                if self._probe_info:
                    msg = f"{msg}\n\n{self._probe_info}"
                raise RuntimeError(msg)
        except Exception as exc:
            self.error_occurred.emit(f"Failed to start audio capture: {exc}")
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()
        self.status_changed.emit(f"Audio capture running ({started_mode})")

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

        self._worker_thread = None

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None

        self._capture_stop.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        self._capture_thread = None
        self._capture_stop.clear()

        if self._sc_recorder is not None:
            try:
                self._sc_recorder.__exit__(None, None, None)
            except Exception:
                pass
            self._sc_recorder = None
        self._sc_candidates = []
        self._sc_candidate_idx = -1
        self._sc_silence_frames = 0

        self._freq_history.clear()
        self._last_output_freq = None
        self._last_confidence = 0.0
        self._missed_updates = 0
        self._pending_jump_freq = None
        self._pending_jump_count = 0

        # Drain remaining chunks.
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self.status_changed.emit("Audio capture stopped")

    def _build_stream_configs(self) -> List[Tuple[str, dict]]:
        if self.capture_mode == "microphone":
            return self._build_microphone_stream_configs()
        return self._build_desktop_stream_configs()

    def _build_microphone_stream_configs(self) -> List[Tuple[str, dict]]:
        blocksize = 1024
        if not self._microphone_sources:
            self.list_microphone_devices(force_refresh=True)

        source = self._find_source_by_id(self.selected_microphone_device_id, self._microphone_sources)
        if source is None and self._microphone_sources:
            source = self._microphone_sources[0]
            self.selected_microphone_device_id = source["id"]

        if source is None:
            default_input = self._safe_default_device(0)
            if default_input is None:
                raise RuntimeError("No microphone input device available")
            source = {
                "label": f"default input #{default_input}",
                "device_index": default_input,
            }

        device_index = int(source["device_index"])
        info = sd.query_devices(device_index)
        channels = max(1, min(2, int(info.get("max_input_channels", 1))))
        samplerate = int(info.get("default_samplerate", 48000))
        return [
            (
                f"Microphone input ({source['label']})",
                {
                    "device": device_index,
                    "samplerate": samplerate,
                    "channels": channels,
                    "dtype": "float32",
                    "blocksize": blocksize,
                    "latency": "low",
                },
            )
        ]

    def _build_desktop_stream_configs(self) -> List[Tuple[str, dict]]:
        if platform.system() != "Windows":
            if not self.allow_microphone_fallback:
                raise RuntimeError(
                    "Desktop capture mode is enabled, but this build supports desktop loopback on Windows only."
                )
            return self._build_microphone_stream_configs()

        if not self._desktop_sources:
            self.list_desktop_devices(force_refresh=True)

        source = self._find_source_by_id(self.selected_desktop_device_id, self._desktop_sources)
        if source is not None:
            return self._build_desktop_configs_from_source(source)
        return self._build_default_desktop_stream_configs()

    def _build_default_desktop_stream_configs(self) -> List[Tuple[str, dict]]:
        blocksize = 1024
        configs: List[Tuple[str, dict]] = []
        hostapis = sd.query_hostapis()
        wasapi_index = self._find_wasapi_hostapi_index(hostapis)
        if wasapi_index is None:
            raise RuntimeError("WASAPI host API not found on this system")

        output_device = self._safe_default_device(1)
        if output_device is None:
            candidate = hostapis[wasapi_index].get("default_output_device")
            if candidate is not None and int(candidate) >= 0:
                output_device = int(candidate)

        if output_device is None:
            raise RuntimeError("No valid default output device found for loopback capture")

        output_info = sd.query_devices(output_device)
        samplerate = int(output_info.get("default_samplerate", 48000))
        output_name = str(output_info.get("name", output_device))
        probe_lines = [f"Default output device: {output_name}"]

        loopback_inputs = self._find_wasapi_loopback_input_devices(wasapi_index, output_info)
        if loopback_inputs:
            loop_names = []
            for loopback_input in loopback_inputs:
                loopback_info = sd.query_devices(loopback_input)
                loop_names.append(str(loopback_info.get("name", loopback_input)))
            probe_lines.append("WASAPI input candidates: " + "; ".join(loop_names))
        else:
            probe_lines.append("WASAPI input candidates: none")

        for loopback_input in loopback_inputs:
            loopback_info = sd.query_devices(loopback_input)
            channels = max(1, min(2, int(loopback_info.get("max_input_channels", 2))))
            configs.append(
                (
                    "WASAPI explicit loopback device",
                    {
                        "device": loopback_input,
                        "samplerate": samplerate,
                        "channels": channels,
                        "dtype": "float32",
                        "blocksize": blocksize,
                        "latency": "low",
                    },
                )
            )

        extra_settings = self._create_wasapi_loopback_settings()
        if extra_settings is not None:
            channels = max(1, min(2, int(output_info.get("max_output_channels", 2))))
            configs.append(
                (
                    "WASAPI output + WasapiSettings(loopback=True)",
                    {
                        "device": output_device,
                        "samplerate": samplerate,
                        "channels": channels,
                        "dtype": "float32",
                        "blocksize": blocksize,
                        "latency": "low",
                        "extra_settings": extra_settings,
                    },
                )
            )

        stereo_mix = self._find_stereo_mix_input_device()
        if stereo_mix is not None:
            sm_info = sd.query_devices(stereo_mix)
            sm_sr = int(sm_info.get("default_samplerate", samplerate))
            channels = max(1, min(2, int(sm_info.get("max_input_channels", 2))))
            configs.append(
                (
                    "Stereo Mix input device",
                    {
                        "device": stereo_mix,
                        "samplerate": sm_sr,
                        "channels": channels,
                        "dtype": "float32",
                        "blocksize": blocksize,
                        "latency": "low",
                    },
                )
            )
            probe_lines.append(f"Stereo Mix candidate: {sm_info.get('name', stereo_mix)}")
        else:
            probe_lines.append("Stereo Mix candidate: none")

        self._probe_info = "\n".join(probe_lines)

        if self.allow_microphone_fallback:
            default_input = self._safe_default_device(0)
            if default_input is not None:
                in_info = sd.query_devices(default_input)
                in_sr = int(in_info.get("default_samplerate", samplerate))
                channels = max(1, min(2, int(in_info.get("max_input_channels", 1))))
                configs.append(
                    (
                        "Default input fallback (microphone)",
                        {
                            "device": default_input,
                            "samplerate": in_sr,
                            "channels": channels,
                            "dtype": "float32",
                            "blocksize": blocksize,
                            "latency": "low",
                        },
                    )
                )

        return configs

    def _build_desktop_configs_from_source(self, source: dict) -> List[Tuple[str, dict]]:
        blocksize = 1024
        source_kind = str(source.get("kind", ""))
        source_label = str(source.get("label", "selected source"))
        device_index = int(source.get("device_index", -1))
        if device_index < 0:
            raise RuntimeError("Invalid selected desktop device")

        if source_kind == "wasapi_output_loopback":
            hostapis = sd.query_hostapis()
            wasapi_index = self._find_wasapi_hostapi_index(hostapis)
            if wasapi_index is None:
                raise RuntimeError("WASAPI host API not found on this system")

            output_info = sd.query_devices(device_index)
            samplerate = int(output_info.get("default_samplerate", 48000))
            configs: List[Tuple[str, dict]] = []

            loopback_inputs = self._find_wasapi_loopback_input_devices(wasapi_index, output_info)
            for loopback_input in loopback_inputs:
                loopback_info = sd.query_devices(loopback_input)
                channels = max(1, min(2, int(loopback_info.get("max_input_channels", 2))))
                configs.append(
                    (
                        f"WASAPI explicit loopback ({source_label})",
                        {
                            "device": loopback_input,
                            "samplerate": samplerate,
                            "channels": channels,
                            "dtype": "float32",
                            "blocksize": blocksize,
                            "latency": "low",
                        },
                    )
                )

            extra_settings = self._create_wasapi_loopback_settings()
            if extra_settings is not None:
                channels = max(1, min(2, int(output_info.get("max_output_channels", 2))))
                configs.append(
                    (
                        f"WASAPI output loopback ({source_label})",
                        {
                            "device": device_index,
                            "samplerate": samplerate,
                            "channels": channels,
                            "dtype": "float32",
                            "blocksize": blocksize,
                            "latency": "low",
                            "extra_settings": extra_settings,
                        },
                    )
                )

            if not configs:
                self._probe_info = f"Selected output has no loopback path: {source_label}"
                return []
            return configs

        info = sd.query_devices(device_index)
        channels = max(1, min(2, int(info.get("max_input_channels", 2))))
        samplerate = int(info.get("default_samplerate", 48000))
        return [
            (
                f"Desktop capture input ({source_label})",
                {
                    "device": device_index,
                    "samplerate": samplerate,
                    "channels": channels,
                    "dtype": "float32",
                    "blocksize": blocksize,
                    "latency": "low",
                },
            )
        ]

    @staticmethod
    def _create_wasapi_loopback_settings():
        """Create WASAPI loopback settings when loopback=True is supported."""
        if not hasattr(sd, "WasapiSettings"):
            return None

        try:
            return sd.WasapiSettings(loopback=True)
        except TypeError:
            return None
        except Exception:
            return None

    @staticmethod
    def _find_wasapi_loopback_input_devices(wasapi_index: int, output_info: dict) -> List[int]:
        """
        Find explicit WASAPI desktop-audio input devices.
        Older sounddevice/portaudio builds expose loopback as dedicated input devices.
        """
        output_name = str(output_info.get("name", "")).lower()
        output_base = output_name.split("(")[0].strip()
        devices = sd.query_devices()

        candidates: List[Tuple[int, int]] = []
        loopback_keywords = [
            "loopback",
            "loop back",
            "回送",
            "环回",
            "回环",
            "回采",
        ]

        for idx, dev in enumerate(devices):
            if int(dev.get("hostapi", -1)) != wasapi_index:
                continue
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue

            name = str(dev.get("name", ""))
            name_lower = name.lower()

            score = 0
            has_loopback_word = any(k in name_lower for k in loopback_keywords)
            if has_loopback_word:
                score += 4
            if output_base and output_base in name_lower:
                score += 2
            if output_name and output_name in name_lower:
                score += 2

            # Keep only likely desktop-capture endpoints, avoid arbitrary microphones.
            if score > 0:
                candidates.append((score, idx))

        candidates.sort(reverse=True)
        return [idx for _, idx in candidates[:4]]

    def _start_soundcard_loopback(self) -> None:
        """
        Fallback desktop capture path using python-soundcard.
        This still captures system output, not microphone input.
        """
        if sc is None:
            raise RuntimeError("python package 'soundcard' is not installed")
        self._sc_candidates = self._resolve_soundcard_loopback_candidates()
        if not self._sc_candidates:
            raise RuntimeError("No soundcard loopback endpoint matched current default speaker")

        samplerate = 48000
        channels = 2
        blocksize = 1024

        open_errors = []
        for idx, mic in enumerate(self._sc_candidates):
            mic_name = str(getattr(mic, "name", mic))
            try:
                self._open_soundcard_recorder(mic, samplerate, channels, blocksize)
                self._sc_candidate_idx = idx
                self.sample_rate = samplerate
                self.channels = channels
                self.status_changed.emit(f"Using soundcard loopback endpoint: {mic_name}")
                self._capture_stop.clear()
                self._capture_thread = threading.Thread(target=self._soundcard_capture_loop, daemon=True)
                self._capture_thread.start()
                return
            except Exception as exc:
                open_errors.append(f"{mic_name}: {exc}")

        raise RuntimeError("Failed to open soundcard loopback endpoints: " + " | ".join(open_errors))

    def _open_soundcard_recorder(self, mic, samplerate: int, channels: int, blocksize: int) -> None:
        if self._sc_recorder is not None:
            try:
                self._sc_recorder.__exit__(None, None, None)
            except Exception:
                pass
            self._sc_recorder = None

        self._sc_recorder = mic.recorder(samplerate=samplerate, channels=channels, blocksize=blocksize)
        self._sc_recorder.__enter__()

    @staticmethod
    def _resolve_soundcard_loopback_candidates() -> List[object]:
        """
        Resolve likely speaker loopback microphones from soundcard.
        Avoid arbitrary physical microphones.
        """
        if sc is None:
            return []

        candidates: List[Tuple[int, object]] = []
        seen_names = set()

        default_speaker = None
        default_speaker_name = ""
        try:
            default_speaker = sc.default_speaker()
            default_speaker_name = str(getattr(default_speaker, "name", ""))
        except Exception:
            pass

        speaker_base = default_speaker_name.lower().split("(")[0].strip()
        speaker_full = default_speaker_name.lower()
        loopback_keywords = ("loopback", "loop back", "回送", "环回", "回环", "stereo mix", "立体声混音")

        def add_candidate(mic, base_score: int = 0):
            name = str(getattr(mic, "name", ""))
            if not name:
                return
            key = name.lower()
            if key in seen_names:
                return
            score = base_score
            if any(k in key for k in loopback_keywords):
                score += 4
            if speaker_base and speaker_base in key:
                score += 3
            if speaker_full and speaker_full in key:
                score += 3
            if "speaker" in key or "output" in key or "扬声器" in key:
                score += 1
            if score <= 0:
                return
            seen_names.add(key)
            candidates.append((score, mic))

        # Best effort: directly map default speaker to loopback microphone.
        if default_speaker_name and hasattr(sc, "get_microphone"):
            try:
                mic = sc.get_microphone(id=default_speaker_name, include_loopback=True)
                if mic is not None:
                    add_candidate(mic, base_score=10)
            except Exception:
                pass

        # API variant 1.
        try:
            mic = sc.default_microphone(include_loopback=True)
            if mic is not None:
                add_candidate(mic, base_score=8)
        except Exception:
            pass

        # API variant 2.
        try:
            all_mics = sc.all_microphones(include_loopback=True)
            for mic in all_mics:
                add_candidate(mic)
        except Exception:
            pass

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in candidates]

    def _soundcard_capture_loop(self) -> None:
        while not self._capture_stop.is_set():
            try:
                data = self._sc_recorder.record(numframes=1024)
                if data is None or len(data) == 0:
                    continue

                if data.ndim == 2 and data.shape[1] > 1:
                    channel_energy = np.mean(data * data, axis=0)
                    mono = data[:, int(np.argmax(channel_energy))]
                else:
                    mono = data[:, 0] if data.ndim == 2 else data

                chunk = np.asarray(mono, dtype=np.float32)
                rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
                if rms < 2e-6:
                    self._sc_silence_frames += 1
                else:
                    self._sc_silence_frames = 0

                # If selected endpoint stays silent for ~3s, try next candidate.
                if self._sc_silence_frames > 140 and self._sc_candidates:
                    if self._try_switch_soundcard_candidate():
                        self._sc_silence_frames = 0
                        continue

                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._audio_queue.put_nowait(chunk)
                    except queue.Full:
                        pass
            except Exception as exc:
                self.status_changed.emit(f"soundcard capture stopped: {exc}")
                break

    def _try_switch_soundcard_candidate(self) -> bool:
        if not self._sc_candidates:
            return False

        samplerate = self.sample_rate or 48000
        channels = self.channels or 2
        blocksize = 1024
        total = len(self._sc_candidates)

        for offset in range(1, total):
            next_idx = (self._sc_candidate_idx + offset) % total
            mic = self._sc_candidates[next_idx]
            mic_name = str(getattr(mic, "name", mic))
            try:
                self._open_soundcard_recorder(mic, samplerate, channels, blocksize)
                self._sc_candidate_idx = next_idx
                self.status_changed.emit(f"Switched soundcard loopback endpoint: {mic_name}")
                return True
            except Exception:
                continue

        return False

    @staticmethod
    def _find_stereo_mix_input_device() -> Optional[int]:
        """Find legacy desktop-capture inputs such as Stereo Mix / What U Hear."""
        best_idx = None
        best_score = -1
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", "")).lower()
            score = 1 if AudioEngine._is_stereo_mix_name(name) else 0
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score <= 0:
            return None
        return best_idx

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.status_changed.emit(f"Audio status: {status}")

        # Vocal-focus stereo fold-down:
        # Pop vocals are usually center-panned, so mid/side helps suppress side instruments.
        if indata.ndim == 2 and indata.shape[1] > 1:
            left = indata[:, 0]
            right = indata[:, 1]
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)

            mid_energy = float(np.mean(mid * mid) + 1e-12)
            side_energy = float(np.mean(side * side) + 1e-12)
            center_ratio = mid_energy / side_energy

            channel_energy = np.mean(indata * indata, axis=0)
            dominant = indata[:, int(np.argmax(channel_energy))]

            if center_ratio >= 1.25:
                mono = mid
            elif center_ratio >= 0.75:
                mono = self.vocal_focus_strength * mid + (1.0 - self.vocal_focus_strength) * dominant
            else:
                # If side content dominates, keep a bit more dominant channel to avoid over-cancel.
                mix = max(0.55, self.vocal_focus_strength - 0.2)
                mono = mix * mid + (1.0 - mix) * dominant
        else:
            mono = indata[:, 0] if indata.ndim == 2 else indata

        chunk = mono.astype(np.float32, copy=False).copy()

        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest chunk when overloaded to keep latency bounded.
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    def _process_loop(self) -> None:
        ring = np.zeros(self.frame_size, dtype=np.float32)
        filled = 0
        samples_since_update = 0
        hop = max(1, int(self.sample_rate * self.update_interval_ms / 1000.0))

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            n = len(chunk)
            if n >= self.frame_size:
                ring[:] = chunk[-self.frame_size :]
                filled = self.frame_size
            else:
                ring = np.roll(ring, -n)
                ring[-n:] = chunk
                filled = min(self.frame_size, filled + n)

            samples_since_update += n
            if filled < self.frame_size or samples_since_update < hop:
                continue

            samples_since_update = 0

            # Hann window reduces spectral leakage and stabilizes pitch estimates.
            frame = ring * self._window
            frame = self._vocal_band_emphasis(frame)
            freq_primary, conf_primary = self.detector.detect_pitch(
                frame,
                self.sample_rate,
                prev_freq=self._last_output_freq,
            )

            freq_secondary = None
            conf_secondary = 0.0
            need_secondary = (
                freq_primary is None
                or conf_primary < 0.55
                or (freq_primary is not None and freq_primary >= 320.0)
            )
            if need_secondary and filled >= self._secondary_frame_size:
                frame2 = ring[-self._secondary_frame_size :] * self._window_secondary
                frame2 = self._vocal_band_emphasis(frame2)
                freq_secondary, conf_secondary = self.detector.detect_pitch(
                    frame2,
                    self.sample_rate,
                    prev_freq=self._last_output_freq,
                )

            freq, confidence = self._merge_multi_resolution(
                freq_primary,
                conf_primary,
                freq_secondary,
                conf_secondary,
            )
            freq, confidence = self._stabilize_frequency(freq, confidence)

            if freq is None:
                result = PitchResult(
                    frequency=None,
                    note_name="--",
                    cents=0.0,
                    confidence=0.0,
                )
            else:
                _, note_name, cents = self.detector.freq_to_note(freq)
                result = PitchResult(
                    frequency=float(freq),
                    note_name=note_name,
                    cents=float(cents),
                    confidence=float(confidence),
                )

            self.pitch_updated.emit(result)

    def _merge_multi_resolution(
        self,
        freq_a: Optional[float],
        conf_a: float,
        freq_b: Optional[float],
        conf_b: float,
    ) -> Tuple[Optional[float], float]:
        """Cross-check two time resolutions and keep only reliable pitch candidates."""
        if freq_a is None and freq_b is None:
            return None, 0.0
        if freq_a is None:
            return (freq_b, conf_b) if conf_b >= self._strict_min_confidence else (None, 0.0)
        if freq_b is None:
            return (freq_a, conf_a) if conf_a >= self._strict_min_confidence else (None, 0.0)

        semitone_diff = abs(12.0 * math.log2(max(freq_a, 1e-9) / max(freq_b, 1e-9)))
        # Relax consensus tolerance in upper register to preserve high-note catch rate.
        if max(freq_a, freq_b) >= self._high_note_threshold_hz:
            consensus_diff = self._consensus_max_semitone_diff + 0.75
        else:
            consensus_diff = self._consensus_max_semitone_diff

        if semitone_diff <= consensus_diff:
            w1 = max(1e-6, conf_a)
            w2 = max(1e-6, conf_b)
            merged = (freq_a * w1 + freq_b * w2) / (w1 + w2)
            return float(merged), float(max(conf_a, conf_b))

        # If two windows disagree, prefer the one closer to current track.
        if self._last_output_freq is not None and self._last_output_freq > 0:
            d1 = abs(12.0 * math.log2(max(freq_a, 1e-9) / self._last_output_freq))
            d2 = abs(12.0 * math.log2(max(freq_b, 1e-9) / self._last_output_freq))
            if d1 < d2:
                return (freq_a, conf_a) if conf_a >= self._strict_min_confidence else (None, 0.0)
            return (freq_b, conf_b) if conf_b >= self._strict_min_confidence else (None, 0.0)

        # No history yet: only trust clearly stronger candidate.
        if conf_a >= conf_b + 0.16 and conf_a >= self._strict_min_confidence:
            return freq_a, conf_a
        if conf_b >= conf_a + 0.16 and conf_b >= self._strict_min_confidence:
            return freq_b, conf_b
        return None, 0.0

    def _stabilize_frequency(self, freq: Optional[float], confidence: float) -> Tuple[Optional[float], float]:
        """
        Improve continuity:
        - reduce octave-flip jumps,
        - median smooth valid detections,
        - hold last valid pitch briefly when one frame is missed.
        """
        if freq is not None:
            min_conf = self._strict_min_confidence
            if freq >= self._high_note_threshold_hz:
                min_conf = max(0.2, min_conf - 0.05)
            if confidence < min_conf:
                freq = None
                confidence = 0.0

        if freq is not None:
            if self._last_output_freq is not None:
                freq = self._reduce_octave_jumps(freq, self._last_output_freq)
                semitone_jump = abs(12.0 * math.log2(max(freq, 1e-9) / max(self._last_output_freq, 1e-9)))
                allow_fast_jump = (
                    freq >= self._high_note_threshold_hz
                    and confidence >= self._high_jump_conf_bypass
                    and semitone_jump <= 12.0
                )

                # In dense accompaniment, reject sudden large jumps unless repeated.
                if semitone_jump > self._max_jump_semitones and not allow_fast_jump and confidence < 0.88:
                    if self._pending_jump_freq is not None:
                        near_pending = abs(12.0 * math.log2(max(freq, 1e-9) / max(self._pending_jump_freq, 1e-9))) < 1.2
                        if near_pending:
                            self._pending_jump_count += 1
                        else:
                            self._pending_jump_freq = float(freq)
                            self._pending_jump_count = 1
                    else:
                        self._pending_jump_freq = float(freq)
                        self._pending_jump_count = 1

                    if self._pending_jump_count < self._jump_confirm_frames:
                        self._missed_updates = 0
                        held_conf = max(0.0, self._last_confidence * 0.92)
                        return self._last_output_freq, held_conf
                else:
                    self._pending_jump_freq = None
                    self._pending_jump_count = 0

            self._freq_history.append(float(freq))
            smoothed = float(np.median(np.array(self._freq_history, dtype=np.float64)))
            self._last_output_freq = smoothed
            self._last_confidence = float(confidence)
            self._missed_updates = 0
            self._pending_jump_freq = None
            self._pending_jump_count = 0
            return smoothed, confidence

        self._missed_updates += 1
        if self._last_output_freq is not None and self._missed_updates <= self._hold_frames:
            held_conf = max(0.0, self._last_confidence * (0.85 ** self._missed_updates))
            return self._last_output_freq, held_conf

        self._freq_history.clear()
        self._last_output_freq = None
        self._last_confidence = 0.0
        self._pending_jump_freq = None
        self._pending_jump_count = 0
        return None, 0.0

    @staticmethod
    def _reduce_octave_jumps(freq: float, ref_freq: float) -> float:
        """Fold to nearest octave around previous value to avoid 2x/0.5x toggling."""
        if ref_freq <= 0:
            return freq

        adjusted = float(freq)
        while adjusted > ref_freq * 1.9:
            adjusted /= 2.0
        while adjusted < ref_freq / 1.9:
            adjusted *= 2.0
        return adjusted

    def _vocal_band_emphasis(self, frame: np.ndarray) -> np.ndarray:
        """
        Keep the typical vocal F0/harmonic zone and attenuate unrelated low/high bands.
        This is a soft emphasis, not a hard gate.
        """
        n = len(frame)
        spec = np.fft.rfft(frame.astype(np.float64, copy=False))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        gain = np.ones_like(freqs)
        # Strongly suppress sub-bass/kick regions.
        gain[freqs < 70.0] = 0.05
        gain[(freqs >= 70.0) & (freqs < 90.0)] = 0.35
        # Emphasize likely vocal fundamental/formant region.
        gain[(freqs >= 90.0) & (freqs <= 260.0)] = 1.22
        gain[(freqs > 260.0) & (freqs <= 500.0)] = 1.32
        gain[(freqs > 500.0) & (freqs <= 1250.0)] = 1.46
        gain[(freqs > 1250.0) & (freqs <= 1800.0)] = 0.95
        # Keep some harmonics, suppress very high bands.
        gain[(freqs > 1800.0) & (freqs <= 3200.0)] = 0.72
        gain[freqs > 3200.0] = 0.15

        # Continuity-guided harmonic emphasis:
        # helps follow vocal line when CD accompaniment is dense.
        prev_f0 = self._last_output_freq
        anchor_strength = float(np.clip((self._last_confidence - 0.42) / 0.4, 0.0, 1.0))
        if prev_f0 is not None and prev_f0 > 0 and anchor_strength > 0.0:
            for harm, amp in ((1, 0.52), (2, 0.38), (3, 0.26), (4, 0.18)):
                center = prev_f0 * harm
                if center >= (self.sample_rate * 0.5 - 20.0):
                    continue
                bandwidth = max(24.0, center * 0.12)
                shape = np.exp(-0.5 * ((freqs - center) / bandwidth) ** 2)
                gain += (amp * anchor_strength) * shape

        filtered = np.fft.irfft(spec * gain, n=n)
        return filtered.astype(np.float32, copy=False)
