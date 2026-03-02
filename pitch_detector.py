import math
from typing import Optional, Tuple

import numpy as np


class PitchDetector:
    """Real-time pitch detector based on a YIN-style CMND function."""

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(
        self,
        min_freq: float = 50.0,
        max_freq: float = 1200.0,
        yin_threshold: float = 0.12,
        min_rms: float = 5e-5,
        max_cmnd_without_threshold_hit: float = 0.45,
        min_confidence: float = 0.0,
    ):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.yin_threshold = yin_threshold
        self.min_rms = min_rms
        self.max_cmnd_without_threshold_hit = max_cmnd_without_threshold_hit
        self.min_confidence = min_confidence

    def detect_pitch(
        self,
        samples: np.ndarray,
        sample_rate: float,
        prev_freq: Optional[float] = None,
    ) -> Tuple[Optional[float], float]:
        """
        Detect fundamental frequency from a mono audio frame.

        Returns:
            (frequency_hz or None, confidence 0..1)
        """
        if samples.ndim != 1:
            raise ValueError("samples must be mono (1-D)")

        if len(samples) < 256:
            return None, 0.0

        x = samples.astype(np.float64, copy=False)
        x = x - np.mean(x)

        # Basic energy gate to avoid unstable output on silence.
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms < self.min_rms:
            return None, 0.0

        # Light pre-emphasis helps vocal harmonics stand out in music mixes.
        x_emph = x.copy()
        x_emph[1:] = x_emph[1:] - 0.97 * x_emph[:-1]

        yin_freq, yin_conf = self._detect_yin(x_emph, sample_rate)
        hps_freq, hps_conf = self._detect_hps(x_emph, sample_rate)

        freq, conf = self._merge_candidates(yin_freq, yin_conf, hps_freq, hps_conf, prev_freq)
        if freq is None:
            return None, 0.0
        if conf < self.min_confidence:
            return None, 0.0
        return freq, conf

    def _detect_yin(self, x: np.ndarray, sample_rate: float) -> Tuple[Optional[float], float]:
        """Time-domain YIN-like detector."""
        n = len(x)
        min_tau = max(2, int(sample_rate / self.max_freq))
        max_tau = min(int(sample_rate / self.min_freq), n - 2)
        if max_tau <= min_tau:
            return None, 0.0

        # Fast autocorrelation via FFT.
        fft_size = 1 << (2 * n - 1).bit_length()
        x_fft = np.fft.rfft(x, n=fft_size)
        acf = np.fft.irfft(x_fft * np.conjugate(x_fft), n=fft_size)[: max_tau + 1].real

        # Difference function d(tau) using autocorrelation identity.
        sq = x * x
        cumsum_sq = np.concatenate(([0.0], np.cumsum(sq)))
        taus = np.arange(1, max_tau + 1)
        s1 = cumsum_sq[n - taus]
        s2 = cumsum_sq[n] - cumsum_sq[taus]

        d = np.zeros(max_tau + 1, dtype=np.float64)
        d[1:] = np.maximum(s1 + s2 - 2.0 * acf[1 : max_tau + 1], 0.0)

        # Cumulative mean normalized difference.
        cmnd = np.ones_like(d)
        cumulative = np.cumsum(d[1:])
        cmnd[1:] = d[1:] * taus / (cumulative + 1e-12)

        # Pick first local minimum under threshold in valid range.
        candidate_tau = None
        search = cmnd[min_tau : max_tau + 1]
        below = np.where(search < self.yin_threshold)[0]
        if below.size > 0:
            tau = int(below[0] + min_tau)
            while tau + 1 <= max_tau and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            candidate_tau = tau
        else:
            tau = int(np.argmin(search) + min_tau)
            # Reject if confidence is too low.
            if cmnd[tau] > self.max_cmnd_without_threshold_hit:
                return None, 0.0
            candidate_tau = tau

        tau_refined = self._parabolic_interpolation(cmnd, candidate_tau)
        if tau_refined <= 0:
            return None, 0.0

        freq = float(sample_rate / tau_refined)
        if not (self.min_freq <= freq <= self.max_freq):
            return None, 0.0

        confidence = float(np.clip(1.0 - cmnd[candidate_tau], 0.0, 1.0))
        return freq, confidence

    def _detect_hps(self, x: np.ndarray, sample_rate: float) -> Tuple[Optional[float], float]:
        """
        Frequency-domain fallback for mixed music (vocal + accompaniment).
        Harmonic Product Spectrum tends to be more robust when fundamental is weak.
        """
        n = len(x)
        if n < 512:
            return None, 0.0

        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(x * window))
        if spectrum.size < 8:
            return None, 0.0

        # Suppress very low frequencies / DC and keep vocal-ish range.
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        valid = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(valid):
            return None, 0.0

        # Smooth a bit to reduce random peaks.
        mag = spectrum.copy()
        if len(mag) > 5:
            mag = np.convolve(mag, np.ones(5) / 5.0, mode="same")

        # HPS with downsampling factors 2..4.
        hps = mag.copy()
        for factor in (2, 3, 4):
            down = mag[::factor]
            hps[: len(down)] *= down

        hps[~valid] = 0.0
        idx = int(np.argmax(hps))
        peak = float(hps[idx])
        if peak <= 0:
            return None, 0.0

        # Relative confidence against median noise floor in valid band.
        band_vals = hps[valid]
        floor = float(np.median(band_vals) + 1e-12)
        conf = float(np.clip((peak / floor - 1.0) / 25.0, 0.0, 1.0))
        if conf < 0.08:
            return None, 0.0

        # Quadratic interpolation around spectral peak.
        f_bin = self._parabolic_interpolation(hps, idx)
        bin_hz = sample_rate / n
        freq = float(f_bin * bin_hz)
        if not (self.min_freq <= freq <= self.max_freq):
            return None, 0.0

        return freq, conf

    def _merge_candidates(
        self,
        yin_freq: Optional[float],
        yin_conf: float,
        hps_freq: Optional[float],
        hps_conf: float,
        prev_freq: Optional[float],
    ) -> Tuple[Optional[float], float]:
        """
        Merge YIN + HPS and prefer continuity for real-time vocal tracking.
        """
        candidates = []
        if yin_freq is not None:
            candidates.append((yin_freq, yin_conf, "yin"))
        if hps_freq is not None:
            candidates.append((hps_freq, hps_conf, "hps"))

        if not candidates:
            return None, 0.0

        # If both exist and are close, average them with confidence weights.
        if len(candidates) == 2:
            f1, c1, _ = candidates[0]
            f2, c2, _ = candidates[1]
            ratio = max(f1, f2) / max(1e-9, min(f1, f2))
            if ratio < 1.08:
                w1 = max(1e-6, c1)
                w2 = max(1e-6, c2)
                f = (f1 * w1 + f2 * w2) / (w1 + w2)
                c = max(c1, c2)
                return float(f), float(c)

        if prev_freq is None or prev_freq <= 0:
            best = max(candidates, key=lambda x: x[1])
            return float(best[0]), float(best[1])

        # Continuity-biased score: confidence minus octave-distance penalty.
        def score(item) -> float:
            f, c, _ = item
            semitone_dist = abs(12.0 * math.log2(max(f, 1e-9) / prev_freq))
            if f >= 349.23 or prev_freq >= 349.23:
                penalty = min(semitone_dist / 22.0, 1.0) * 0.42
            else:
                penalty = min(semitone_dist / 18.0, 1.0) * 0.5
            return c - penalty

        best = max(candidates, key=score)
        return float(best[0]), float(best[1])

    @staticmethod
    def _parabolic_interpolation(arr: np.ndarray, idx: int) -> float:
        """Refine peak/valley position by fitting a parabola around idx."""
        if idx <= 0 or idx >= len(arr) - 1:
            return float(idx)

        y0 = arr[idx - 1]
        y1 = arr[idx]
        y2 = arr[idx + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) < 1e-12:
            return float(idx)
        shift = 0.5 * (y0 - y2) / denom
        shift = float(np.clip(shift, -1.0, 1.0))
        return float(idx + shift)

    @classmethod
    def freq_to_note(cls, freq: float) -> Tuple[int, str, float]:
        """
        Convert frequency to equal temperament pitch.

        Returns:
            (midi_note, note_name, cents_offset)
        """
        midi_float = 69.0 + 12.0 * math.log2(freq / 440.0)
        midi_rounded = int(round(midi_float))
        cents = (midi_float - midi_rounded) * 100.0

        note_name = cls.NOTE_NAMES[midi_rounded % 12]
        octave = midi_rounded // 12 - 1
        full_name = f"{note_name}{octave}"

        return midi_rounded, full_name, cents
