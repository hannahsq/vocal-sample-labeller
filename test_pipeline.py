"""
Audio Labeling Pipeline — Unit Test Suite
==========================================

Each test class covers one metric family.  Tests use dedicated synthetic WAV
files so failures are unambiguous.  The suite runs with unittest and also
prints a human-readable pass/fail summary table at the end.

Run:
    python test_pipeline.py
    python -m pytest test_pipeline.py -v   (if pytest is available)
"""

import wave
import os
import sys
import math
import unittest
from typing import List

import numpy as np
from scipy.signal import butter, lfilter as sp_lfilter

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (
    PipelineConfig,
    loudness_dbfs,
    jitter_rap,
    shimmer_local,
    process_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16000
DURATION = 0.5

_tmpdir = "/tmp/pipeline_tests"
os.makedirs(_tmpdir, exist_ok=True)


def _write_wav(filename: str, samples: np.ndarray, sr: int = SR) -> str:
    """Write a float32 array (±1.0) as 16-bit mono WAV. Returns full path."""
    path = os.path.join(_tmpdir, filename)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_int = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int.tobytes())
    return path


def _sine(freq: float, amp: float = 1.0, dur: float = DURATION, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _white_noise(amp: float = 1.0, dur: float = DURATION, sr: int = SR,
                 seed: int = 42) -> np.ndarray:
    """Uniform white noise peak-normalised to `amp`.
    Note: RMS of peak-normalised uniform noise ≈ amp/√3 ≈ 0.577·amp,
    which gives loudness ≈ -4.77 dBFS for amp=1.0.
    """
    rng = np.random.default_rng(seed)
    sig = rng.uniform(-1.0, 1.0, int(sr * dur)).astype(np.float32)
    sig = sig / np.max(np.abs(sig))   # peak-normalise
    return (amp * sig).astype(np.float32)


def _silence(dur: float = DURATION, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * dur), dtype=np.float32)


def _bandpass_noise(center: float, bw: float = 100.0, dur: float = 1.0,
                    sr: int = SR, seed: int = 11) -> np.ndarray:
    """White noise band-passed around center_hz using a 4th-order Butterworth."""
    nyq  = sr / 2.0
    low  = max(0.01, (center - bw / 2) / nyq)
    high = min(0.99, (center + bw / 2) / nyq)
    b, a = butter(4, [low, high], btype="band")
    noise = _white_noise(dur=dur, sr=sr, seed=seed)
    sig   = sp_lfilter(b, a, noise).astype(np.float32)
    peak  = np.max(np.abs(sig))
    return (sig / peak if peak > 1e-6 else sig).astype(np.float32)


def _cfg(**kwargs) -> PipelineConfig:
    defaults = dict(
        target_sr=0, window_ms=25.0, hop_ms=10.0,
        f0_min=60, f0_max=400, voicing_threshold=0.45, n_formants=7,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def _median_non_nan(vals: List[float]) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return float(np.median(valid)) if valid else float("nan")


def _all_nan(vals: List[float]) -> bool:
    return all(math.isnan(v) for v in vals)


# ---------------------------------------------------------------------------
# Test result registry — feeds the summary table
# ---------------------------------------------------------------------------

_results: List[dict] = []


def _record(test_name: str, passed: bool, detail: str = ""):
    _results.append({"test": test_name, "passed": passed, "detail": detail})


# ---------------------------------------------------------------------------
# 1. LOUDNESS
# ---------------------------------------------------------------------------

class TestLoudness(unittest.TestCase):

    def test_silence_at_or_below_minus100_dbfs(self):
        """Pure silence → all windows ≤ -100 dBFS (pipeline floor value)."""
        path = _write_wav("silence.wav", _silence())
        vals = [r.loudness_dbfs for r in process_file(path, _cfg())]
        try:
            self.assertGreater(len(vals), 0, "No windows produced from silence file")
            for v in vals:
                self.assertLessEqual(v, -100.0,
                    f"Silence: expected ≤ -100 dBFS, got {v:.2f}")
            _record("silence loudness ≤ -100 dBFS", True)
        except AssertionError as e:
            _record("silence loudness ≤ -100 dBFS", False, str(e)); raise

    def test_full_amplitude_white_noise_loudness_in_expected_range(self):
        """Amplitude-1.0 peak-normalised white noise → loudness ≈ -4.77 dBFS.

        Uniform white noise has RMS = peak/√3, so loudness = 20·log10(1/√3) ≈ -4.77 dBFS.
        Tested as range [-6, -3] dBFS.
        """
        path = _write_wav("noise_amp1.wav", _white_noise(amp=1.0))
        med  = _median_non_nan([r.loudness_dbfs for r in process_file(path, _cfg())])
        try:
            self.assertGreater(med, -6.0,
                f"Amp-1 noise: expected > -6 dBFS, got {med:.2f}")
            self.assertLess(med, -3.0,
                f"Amp-1 noise: expected < -3 dBFS (theory ≈ -4.77), got {med:.2f}")
            _record("noise amp-1 loudness in [-6, -3] dBFS", True, f"median={med:.3f} dBFS")
        except AssertionError as e:
            _record("noise amp-1 loudness in [-6, -3] dBFS", False, str(e)); raise

    def test_full_amplitude_sine_loudness_near_minus3_dbfs(self):
        """Amplitude-1.0 sine wave → loudness ≈ -3.01 dBFS (RMS = 1/√2).
        Tested as range [-4, -2] dBFS.
        """
        path = _write_wav("sine_amp1.wav", _sine(220, amp=1.0))
        med  = _median_non_nan([r.loudness_dbfs for r in process_file(path, _cfg())])
        try:
            self.assertGreater(med, -4.0,
                f"Amp-1 sine: expected > -4 dBFS, got {med:.2f}")
            self.assertLess(med, -2.0,
                f"Amp-1 sine: expected < -2 dBFS (≈ -3.01 theoretical), got {med:.2f}")
            _record("sine amp-1 loudness in [-4, -2] dBFS", True,
                    f"median={med:.3f} dBFS (theory=-3.01)")
        except AssertionError as e:
            _record("sine amp-1 loudness in [-4, -2] dBFS", False, str(e)); raise

    def test_half_amplitude_sine_loudness_near_minus9_dbfs(self):
        """Amplitude-0.5 sine wave → loudness ≈ -9.03 dBFS.

        Halving amplitude reduces loudness by 6.02 dB:
            20·log10(0.5/√2) = 20·log10(1/√2) + 20·log10(0.5) = -3.01 - 6.02 = -9.03 dBFS
        Tested as range [-9.5, -8.5] dBFS.

        Note: the range [-6.5, -5.5] dBFS would apply to a signal whose *RMS* equals 0.5
        (e.g. a full-scale square wave attenuated to 0.5), not a 0.5-amplitude sine.
        """
        path = _write_wav("sine_amp05.wav", _sine(220, amp=0.5))
        med  = _median_non_nan([r.loudness_dbfs for r in process_file(path, _cfg())])
        try:
            self.assertGreater(med, -9.5,
                f"Amp-0.5 sine: expected > -9.5 dBFS, got {med:.2f}")
            self.assertLess(med, -8.5,
                f"Amp-0.5 sine: expected < -8.5 dBFS (≈ -9.03 theoretical), got {med:.2f}")
            _record("sine amp-0.5 loudness in [-9.5, -8.5] dBFS", True,
                    f"median={med:.3f} dBFS (theory=-9.03)")
        except AssertionError as e:
            _record("sine amp-0.5 loudness in [-9.5, -8.5] dBFS", False, str(e)); raise

    def test_louder_signal_has_higher_dbfs(self):
        """Monotonicity: amp-1.0 signal is louder than amp-0.5 signal."""
        med1 = _median_non_nan([r.loudness_dbfs for r in process_file(
            _write_wav("sine_amp1_m.wav", _sine(220, 1.0)), _cfg())])
        med2 = _median_non_nan([r.loudness_dbfs for r in process_file(
            _write_wav("sine_amp05_m.wav", _sine(220, 0.5)), _cfg())])
        try:
            self.assertGreater(med1, med2,
                f"amp-1 ({med1:.2f}) should be louder than amp-0.5 ({med2:.2f})")
            _record("loudness monotonicity: amp-1 > amp-0.5", True,
                    f"amp-1={med1:.2f}, amp-0.5={med2:.2f} dBFS")
        except AssertionError as e:
            _record("loudness monotonicity: amp-1 > amp-0.5", False, str(e)); raise

    def test_loudness_dbfs_exact_value(self):
        """Unit test: loudness_dbfs() on a flat 0.1 signal = exactly -20 dBFS."""
        sig    = np.full(SR, 0.1, dtype=np.float32)
        result = loudness_dbfs(sig)
        try:
            self.assertAlmostEqual(result, -20.0, places=3,
                msg=f"Expected -20.0 dBFS for flat 0.1 signal, got {result:.4f}")
            _record("loudness_dbfs() exact: flat 0.1 → -20 dBFS", True)
        except AssertionError as e:
            _record("loudness_dbfs() exact: flat 0.1 → -20 dBFS", False, str(e)); raise

    def test_noise_louder_than_half_amplitude_noise(self):
        """amp-1.0 noise louder than amp-0.5 noise (monotonicity for noise signals)."""
        med1 = _median_non_nan([r.loudness_dbfs for r in process_file(
            _write_wav("noise_amp1_m.wav", _white_noise(1.0)), _cfg())])
        med2 = _median_non_nan([r.loudness_dbfs for r in process_file(
            _write_wav("noise_amp05_m.wav", _white_noise(0.5)), _cfg())])
        try:
            self.assertGreater(med1, med2,
                f"amp-1 noise ({med1:.2f}) should be louder than amp-0.5 noise ({med2:.2f})")
            _record("loudness monotonicity: noise amp-1 > noise amp-0.5", True,
                    f"amp-1={med1:.2f}, amp-0.5={med2:.2f} dBFS")
        except AssertionError as e:
            _record("loudness monotonicity: noise amp-1 > noise amp-0.5", False, str(e)); raise


# ---------------------------------------------------------------------------
# 2. PERIODICITY
# ---------------------------------------------------------------------------

class TestPeriodicity(unittest.TestCase):

    def test_white_noise_periodicity_below_0p2(self):
        """White noise → median periodicity < 0.2 (flat autocorrelation)."""
        path = _write_wav("noise_per.wav", _white_noise())
        med  = _median_non_nan([r.periodicity for r in
                                 process_file(path, _cfg(voicing_threshold=0.0))])
        try:
            self.assertLess(med, 0.2,
                f"Noise: expected periodicity < 0.2, got {med:.3f}")
            _record("noise periodicity < 0.2", True, f"median={med:.3f}")
        except AssertionError as e:
            _record("noise periodicity < 0.2", False, str(e)); raise

    def test_sine_periodicity_above_0p75(self):
        """Pure 200 Hz sine → median periodicity > 0.75.

        Note: for a 25 ms Hamming-windowed frame at 200 Hz (exactly 5 cycles),
        the normalised autocorrelation at lag T is cos(2π·T/T_window) = cos(2π/5)
        ≈ 0.80 — this is the theoretical maximum, not 1.0, because the window
        reduces correlation at longer lags. 0.75 is a robust lower bound.
        """
        path = _write_wav("sine_per.wav", _sine(200, amp=1.0))
        med  = _median_non_nan([r.periodicity for r in
                                 process_file(path, _cfg(voicing_threshold=0.0))])
        try:
            self.assertGreater(med, 0.75,
                f"Sine: expected periodicity > 0.75, got {med:.3f}")
            _record("sine periodicity > 0.75", True, f"median={med:.3f}")
        except AssertionError as e:
            _record("sine periodicity > 0.75", False, str(e)); raise

    def test_sine_periodicity_much_higher_than_noise(self):
        """Sine periodicity is at least 3× the noise periodicity."""
        cfg       = _cfg(voicing_threshold=0.0)
        med_noise = _median_non_nan([r.periodicity for r in
                                      process_file(_write_wav("noise_cmp.wav", _white_noise()), cfg)])
        med_sine  = _median_non_nan([r.periodicity for r in
                                      process_file(_write_wav("sine_cmp.wav", _sine(200)), cfg)])
        try:
            self.assertGreater(med_sine, med_noise * 3,
                f"Sine ({med_sine:.3f}) should be > 3× noise ({med_noise:.3f})")
            _record("sine periodicity > 3× noise periodicity", True,
                    f"sine={med_sine:.3f}, noise={med_noise:.3f}")
        except AssertionError as e:
            _record("sine periodicity > 3× noise periodicity", False, str(e)); raise

    def test_mixed_signal_between_noise_and_sine(self):
        """50/50 sine+noise mix: noise < mix < sine (graded check)."""
        t   = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
        rng = np.random.default_rng(7)
        mix = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.5 * rng.uniform(-1, 1, len(t))
        mix = (mix / np.max(np.abs(mix))).astype(np.float32)

        cfg = _cfg(voicing_threshold=0.0)
        def med(sig, fname):
            return _median_non_nan([r.periodicity for r in
                                     process_file(_write_wav(fname, sig), cfg)])
        m_noise = med(_white_noise(seed=7), "mix_noise.wav")
        m_mix   = med(mix,                  "mix_mixed.wav")
        m_sine  = med(_sine(200),           "mix_sine.wav")
        try:
            self.assertGreater(m_mix, m_noise,
                f"Mixed ({m_mix:.3f}) should be > noise ({m_noise:.3f})")
            self.assertLess(m_mix, m_sine,
                f"Mixed ({m_mix:.3f}) should be < sine ({m_sine:.3f})")
            _record("periodicity: noise < mixed < sine", True,
                    f"noise={m_noise:.3f}, mixed={m_mix:.3f}, sine={m_sine:.3f}")
        except AssertionError as e:
            _record("periodicity: noise < mixed < sine", False, str(e)); raise

    def test_periodicity_bounded_0_to_1(self):
        """Periodicity is always in [0, 1] for silence, noise and sine."""
        cfg     = _cfg(voicing_threshold=0.0)
        sources = [("silence", _silence()), ("noise", _white_noise()), ("sine", _sine(150))]
        try:
            for name, sig in sources:
                for r in process_file(_write_wav(f"bounds_{name}.wav", sig), cfg):
                    self.assertGreaterEqual(r.periodicity, 0.0,
                        f"{name}: periodicity {r.periodicity} < 0")
                    self.assertLessEqual(r.periodicity, 1.0,
                        f"{name}: periodicity {r.periodicity} > 1")
            _record("periodicity always in [0, 1]", True)
        except AssertionError as e:
            _record("periodicity always in [0, 1]", False, str(e)); raise


# ---------------------------------------------------------------------------
# 3. JITTER
# ---------------------------------------------------------------------------

class TestJitter(unittest.TestCase):

    @staticmethod
    def _fm_sine(base: float, depth: float, mod: float = 8.0,
                 dur: float = 1.0) -> np.ndarray:
        """Sine with sinusoidal frequency modulation → elevated jitter."""
        t     = np.arange(int(SR * dur)) / SR
        phase = 2 * np.pi * (base * t + depth / mod * np.sin(2 * np.pi * mod * t))
        return np.sin(phase).astype(np.float32)

    def test_clean_sine_jitter_below_1_pct(self):
        """Perfect sine → median RAP jitter < 1% on voiced frames."""
        path = _write_wav("jitter_clean.wav", _sine(200, dur=1.0))
        vals = [r.jitter_local for r in process_file(path, _cfg(voicing_threshold=0.45))
                if not math.isnan(r.jitter_local)]
        try:
            self.assertGreater(len(vals), 0, "No voiced frames detected for clean sine")
            med = float(np.median(vals))
            self.assertLess(med, 1.0,
                f"Clean sine jitter should be < 1%, got median {med:.3f}%")
            _record("clean sine jitter < 1%", True, f"median={med:.3f}%")
        except AssertionError as e:
            _record("clean sine jitter < 1%", False, str(e)); raise

    def test_fm_sine_jitter_higher_than_clean(self):
        """FM-modulated sine has clearly higher jitter than clean sine."""
        cfg = _cfg(voicing_threshold=0.45)
        def med_jitter(path):
            vals = [r.jitter_local for r in process_file(path, cfg)
                    if not math.isnan(r.jitter_local)]
            return float(np.median(vals)) if vals else float("nan")

        j_clean = med_jitter(_write_wav("jitter_clean2.wav", _sine(200, dur=1.0)))
        j_fm    = med_jitter(_write_wav("jitter_fm.wav", self._fm_sine(200, depth=10.0)))
        try:
            self.assertFalse(math.isnan(j_clean), "No voiced frames in clean sine")
            self.assertFalse(math.isnan(j_fm),    "No voiced frames in FM sine")
            self.assertGreater(j_fm, j_clean,
                f"FM jitter ({j_fm:.2f}%) should exceed clean ({j_clean:.2f}%)")
            _record("FM sine jitter > clean sine jitter", True,
                    f"clean={j_clean:.3f}%, FM={j_fm:.3f}%")
        except AssertionError as e:
            _record("FM sine jitter > clean sine jitter", False, str(e)); raise

    def test_jitter_nan_for_unvoiced_frames(self):
        """Silence and noise at very high voicing threshold → all NaN jitter."""
        cfg = _cfg(voicing_threshold=0.99)
        for name, sig in [("silence", _silence()), ("noise", _white_noise())]:
            vals = [r.jitter_local for r in
                    process_file(_write_wav(f"jitter_uv_{name}.wav", sig), cfg)]
            try:
                self.assertTrue(_all_nan(vals),
                    f"Expected all NaN jitter for {name}")
                _record(f"jitter NaN for unvoiced {name}", True)
            except AssertionError as e:
                _record(f"jitter NaN for unvoiced {name}", False, str(e)); raise

    def test_jitter_rap_unit_equal_periods(self):
        """Unit test: jitter_rap() on perfectly equal periods → 0%."""
        result = jitter_rap([0.01] * 10)
        try:
            self.assertAlmostEqual(result, 0.0, places=6)
            _record("jitter_rap(): equal periods → 0%", True)
        except AssertionError as e:
            _record("jitter_rap(): equal periods → 0%", False, str(e)); raise

    def test_jitter_rap_unit_large_perturbation(self):
        """Unit test: jitter_rap() on alternating periods → jitter > 10%."""
        result = jitter_rap([0.01, 0.015, 0.01, 0.015, 0.01, 0.015])
        try:
            self.assertGreater(result, 10.0,
                f"Large perturbation should give > 10%, got {result:.2f}%")
            _record("jitter_rap(): large perturbation > 10%", True, f"{result:.2f}%")
        except AssertionError as e:
            _record("jitter_rap(): large perturbation > 10%", False, str(e)); raise

    def test_jitter_rap_unit_too_few_periods(self):
        """Unit test: jitter_rap() with < 3 periods → NaN (insufficient data)."""
        self.assertTrue(math.isnan(jitter_rap([])))
        self.assertTrue(math.isnan(jitter_rap([0.01])))
        self.assertTrue(math.isnan(jitter_rap([0.01, 0.01])))
        _record("jitter_rap(): < 3 periods → NaN", True)


# ---------------------------------------------------------------------------
# 4. SHIMMER
# ---------------------------------------------------------------------------

class TestShimmer(unittest.TestCase):

    @staticmethod
    def _am_sine(freq: float, depth: float, mod: float = 5.0,
                 dur: float = 1.0) -> np.ndarray:
        """Amplitude-modulated sine → elevated shimmer."""
        t   = np.linspace(0, dur, int(SR * dur), endpoint=False)
        env = 1.0 + depth * np.sin(2 * np.pi * mod * t)
        sig = env * np.sin(2 * np.pi * freq * t)
        return (sig / np.max(np.abs(sig))).astype(np.float32)

    def test_clean_sine_shimmer_below_2_pct(self):
        """Perfect sine → median shimmer < 2% on voiced frames."""
        path = _write_wav("shimmer_clean.wav", _sine(200, dur=1.0))
        vals = [r.shimmer_local for r in process_file(path, _cfg(voicing_threshold=0.45))
                if not math.isnan(r.shimmer_local)]
        try:
            self.assertGreater(len(vals), 0, "No voiced frames for clean sine")
            med = float(np.median(vals))
            self.assertLess(med, 2.0,
                f"Clean sine shimmer should be < 2%, got {med:.3f}%")
            _record("clean sine shimmer < 2%", True, f"median={med:.3f}%")
        except AssertionError as e:
            _record("clean sine shimmer < 2%", False, str(e)); raise

    def test_am_sine_shimmer_higher_than_clean(self):
        """AM-modulated sine has clearly higher shimmer than clean sine."""
        cfg = _cfg(voicing_threshold=0.45)
        def med_shimmer(path):
            vals = [r.shimmer_local for r in process_file(path, cfg)
                    if not math.isnan(r.shimmer_local)]
            return float(np.median(vals)) if vals else float("nan")

        s_clean = med_shimmer(_write_wav("shimmer_clean2.wav", _sine(200, dur=1.0)))
        s_am    = med_shimmer(_write_wav("shimmer_am.wav", self._am_sine(200, depth=0.5)))
        try:
            self.assertFalse(math.isnan(s_clean), "No voiced frames in clean sine")
            self.assertFalse(math.isnan(s_am),    "No voiced frames in AM sine")
            self.assertGreater(s_am, s_clean,
                f"AM shimmer ({s_am:.2f}%) should exceed clean ({s_clean:.2f}%)")
            _record("AM sine shimmer > clean sine shimmer", True,
                    f"clean={s_clean:.3f}%, AM={s_am:.3f}%")
        except AssertionError as e:
            _record("AM sine shimmer > clean sine shimmer", False, str(e)); raise

    def test_shimmer_nan_for_unvoiced_frames(self):
        """Silence and noise at very high voicing threshold → all NaN shimmer."""
        cfg = _cfg(voicing_threshold=0.99)
        for name, sig in [("silence", _silence()), ("noise", _white_noise())]:
            vals = [r.shimmer_local for r in
                    process_file(_write_wav(f"shimmer_uv_{name}.wav", sig), cfg)]
            try:
                self.assertTrue(_all_nan(vals),
                    f"Expected all NaN shimmer for {name}")
                _record(f"shimmer NaN for unvoiced {name}", True)
            except AssertionError as e:
                _record(f"shimmer NaN for unvoiced {name}", False, str(e)); raise

    def test_shimmer_local_unit_equal_amps(self):
        """Unit test: shimmer_local() on equal amplitudes → 0%."""
        result = shimmer_local([1.0] * 8)
        try:
            self.assertAlmostEqual(result, 0.0, places=6)
            _record("shimmer_local(): equal amps → 0%", True)
        except AssertionError as e:
            _record("shimmer_local(): equal amps → 0%", False, str(e)); raise

    def test_shimmer_local_unit_alternating_amps(self):
        """Unit test: shimmer_local() on alternating 0.5/1.0 amps → shimmer > 50%."""
        result = shimmer_local([0.5, 1.0] * 4)
        try:
            self.assertGreater(result, 50.0,
                f"Alternating amps should give > 50%, got {result:.2f}%")
            _record("shimmer_local(): alternating 0.5/1.0 amps > 50%", True,
                    f"{result:.2f}%")
        except AssertionError as e:
            _record("shimmer_local(): alternating 0.5/1.0 amps > 50%", False, str(e)); raise

    def test_shimmer_local_unit_too_few_amps(self):
        """Unit test: shimmer_local() with < 2 amplitude values → NaN."""
        self.assertTrue(math.isnan(shimmer_local([])))
        self.assertTrue(math.isnan(shimmer_local([1.0])))
        _record("shimmer_local(): < 2 amps → NaN", True)


# ---------------------------------------------------------------------------
# 5. FORMANTS
# ---------------------------------------------------------------------------

class TestFormants(unittest.TestCase):

    def test_pure_sine_produces_exactly_one_formant(self):
        """A pure sine wave is modelled by LPC as a single very-narrow-bandwidth
        pole, so every window should have exactly F1 present and F2–F7 all NaN."""
        path    = _write_wav("fmt_sine_one.wav", _sine(200, dur=1.0))
        results = process_file(path, _cfg(n_formants=7))
        try:
            for r in results:
                f1 = r.F1
                self.assertFalse(math.isnan(f1),
                    "Sine should always produce exactly F1")
                for i in range(2, 8):
                    fi = getattr(r, f"F{i}")
                    self.assertTrue(math.isnan(fi),
                        f"Sine should have NaN for F{i}, got {fi:.1f} Hz")
            # F1 should be near the sine frequency
            f1_vals = [r.F1 for r in results]
            med_f1  = float(np.median(f1_vals))
            self.assertGreater(med_f1, 150, f"F1 ({med_f1:.0f} Hz) too low for 200 Hz sine")
            self.assertLess(med_f1,    250, f"F1 ({med_f1:.0f} Hz) too high for 200 Hz sine")
            _record("pure sine: exactly 1 formant (F1) near sine frequency", True,
                    f"F1={med_f1:.0f} Hz, F2–F7 all NaN")
        except AssertionError as e:
            _record("pure sine: exactly 1 formant (F1) near sine frequency", False, str(e)); raise

    def test_white_noise_f1_highly_variable(self):
        """White noise has a flat spectrum → LPC finds poles at random frequencies.
        F1 standard deviation should be very high (> 500 Hz)."""
        path    = _write_wav("fmt_noise_var.wav", _white_noise(dur=1.0))
        results = process_file(path, _cfg(n_formants=7))
        f1_vals = [r.F1 for r in results if not math.isnan(r.F1)]
        try:
            self.assertGreater(len(f1_vals), len(results) * 0.5,
                "Expected F1 present in >50% of noise windows")
            std_f1 = float(np.std(f1_vals))
            self.assertGreater(std_f1, 500,
                f"Noise F1 std ({std_f1:.0f} Hz) should be > 500 Hz (no stable resonance)")
            _record("white noise F1 std > 500 Hz (unstable)", True,
                    f"std={std_f1:.0f} Hz")
        except AssertionError as e:
            _record("white noise F1 std > 500 Hz (unstable)", False, str(e)); raise

    def test_bandpass_noise_f1_stable_near_800hz(self):
        """Bandpass noise at 800 Hz → F1 clusters stably near 800 Hz (±200 Hz).
        F1 std should be much lower than for flat white noise."""
        path    = _write_wav("fmt_bp800.wav", _bandpass_noise(800, bw=120, dur=1.0))
        results = process_file(path, _cfg(n_formants=7, lpc_order=16))
        f1_vals = [r.F1 for r in results if not math.isnan(r.F1)]
        try:
            self.assertGreater(len(f1_vals), len(results) * 0.5,
                "Expected F1 in >50% of windows for bandpass noise")
            med_f1 = float(np.median(f1_vals))
            std_f1 = float(np.std(f1_vals))
            self.assertGreater(med_f1, 600, f"F1 median ({med_f1:.0f} Hz) below 600 Hz")
            self.assertLess(med_f1,    1000, f"F1 median ({med_f1:.0f} Hz) above 1000 Hz")
            self.assertLess(std_f1,    200,  f"F1 std ({std_f1:.0f} Hz) should be < 200 Hz")
            _record("bandpass 800 Hz: F1 stable within ±200 Hz", True,
                    f"median={med_f1:.0f} Hz, std={std_f1:.0f} Hz")
        except AssertionError as e:
            _record("bandpass 800 Hz: F1 stable within ±200 Hz", False, str(e)); raise

    def test_two_resonances_recovered_as_f1_f2(self):
        """Two bandpass resonances at 500 & 1500 Hz → F1≈500 Hz, F2≈1500 Hz."""
        bp500  = _bandpass_noise(500,  bw=100, dur=1.0, seed=21)
        bp1500 = _bandpass_noise(1500, bw=100, dur=1.0, seed=22)
        sig    = bp500 + bp1500
        sig    = (sig / np.max(np.abs(sig))).astype(np.float32)
        path   = _write_wav("fmt_two.wav", sig)
        results = process_file(path, _cfg(n_formants=7, lpc_order=20))
        f1_vals = [r.F1 for r in results if not math.isnan(r.F1)]
        f2_vals = [r.F2 for r in results if not math.isnan(r.F2)]
        try:
            self.assertGreater(len(f1_vals), len(results) * 0.5)
            self.assertGreater(len(f2_vals), len(results) * 0.5)
            m1, m2 = float(np.median(f1_vals)), float(np.median(f2_vals))
            self.assertGreater(m1,  300, f"F1 ({m1:.0f}) below 300 Hz")
            self.assertLess(m1,     700, f"F1 ({m1:.0f}) above 700 Hz")
            self.assertGreater(m2, 1200, f"F2 ({m2:.0f}) below 1200 Hz")
            self.assertLess(m2,    1800, f"F2 ({m2:.0f}) above 1800 Hz")
            _record("two-resonance: F1≈500 Hz, F2≈1500 Hz", True,
                    f"F1={m1:.0f} Hz, F2={m2:.0f} Hz")
        except AssertionError as e:
            _record("two-resonance: F1≈500 Hz, F2≈1500 Hz", False, str(e)); raise

    def test_formant_frequencies_always_ascending(self):
        """F1 ≤ F2 ≤ ... must hold in every window where multiple formants are present."""
        sig     = _bandpass_noise(800, bw=300, dur=1.0)
        results = process_file(_write_wav("fmt_order.wav", sig),
                               _cfg(n_formants=7, lpc_order=20))
        try:
            for r in results:
                present = [getattr(r, f"F{i}") for i in range(1, 8)
                           if not math.isnan(getattr(r, f"F{i}"))]
                for a, b in zip(present, present[1:]):
                    self.assertLessEqual(a, b, f"Non-ascending formants: {present}")
            _record("formant frequencies always ascending", True)
        except AssertionError as e:
            _record("formant frequencies always ascending", False, str(e)); raise

    def test_formant_bandwidths_always_positive(self):
        """All reported bandwidths B1–B7 must be > 0 Hz."""
        sig     = _bandpass_noise(800, bw=200, dur=1.0)
        results = process_file(_write_wav("fmt_bw.wav", sig),
                               _cfg(n_formants=7, lpc_order=20))
        try:
            for r in results:
                for i in range(1, 8):
                    bw = getattr(r, f"B{i}")
                    if not math.isnan(bw):
                        self.assertGreater(bw, 0.0, f"B{i} ≤ 0: {bw}")
            _record("formant bandwidths always positive", True)
        except AssertionError as e:
            _record("formant bandwidths always positive", False, str(e)); raise

    def test_formant_count_respects_n_formants_cap(self):
        """Reported formants never exceed the configured n_formants cap."""
        sig = _bandpass_noise(800, bw=400, dur=1.0)
        for cap in [2, 4, 7]:
            results = process_file(_write_wav(f"fmt_cap{cap}.wav", sig),
                                   _cfg(n_formants=cap, lpc_order=20))
            try:
                for r in results:
                    count = sum(1 for i in range(1, 8)
                                if not math.isnan(getattr(r, f"F{i}")))
                    self.assertLessEqual(count, cap,
                        f"cap={cap}: got {count} formants")
                _record(f"formant count ≤ n_formants={cap}", True)
            except AssertionError as e:
                _record(f"formant count ≤ n_formants={cap}", False, str(e)); raise

    def test_formant_freq_bandwidth_nan_paired(self):
        """If Fn is NaN then Bn must also be NaN (and vice versa)."""
        sig     = _bandpass_noise(800, bw=200, dur=1.0)
        results = process_file(_write_wav("fmt_pair.wav", sig),
                               _cfg(n_formants=7, lpc_order=20))
        try:
            for r in results:
                for i in range(1, 8):
                    fn_nan = math.isnan(getattr(r, f"F{i}"))
                    bn_nan = math.isnan(getattr(r, f"B{i}"))
                    self.assertEqual(fn_nan, bn_nan,
                        f"F{i}/B{i} NaN mismatch: F{i}={getattr(r,'F'+str(i))}, "
                        f"B{i}={getattr(r,'B'+str(i))}")
            _record("F/B NaN always paired", True)
        except AssertionError as e:
            _record("F/B NaN always paired", False, str(e)); raise


# ---------------------------------------------------------------------------
# 6. PIPELINE INTEGRITY
# ---------------------------------------------------------------------------

class TestPipelineIntegrity(unittest.TestCase):

    def test_window_count_correct(self):
        """Number of windows = floor((N - win_samples) / hop_samples) + 1."""
        sig  = _sine(200, dur=1.0)
        path = _write_wav("wc_count.wav", sig)
        results  = process_file(path, _cfg(window_ms=25.0, hop_ms=10.0))
        n        = len(sig)
        win_s    = int(round(0.025 * SR))
        hop_s    = int(round(0.010 * SR))
        expected = (n - win_s) // hop_s + 1
        try:
            self.assertEqual(len(results), expected,
                f"Expected {expected} windows, got {len(results)}")
            _record("window count formula correct", True,
                    f"expected={expected}, got={len(results)}")
        except AssertionError as e:
            _record("window count formula correct", False, str(e)); raise

    def test_window_t_start_monotonic_by_hop(self):
        """t_start increases by exactly hop_ms between consecutive windows."""
        path    = _write_wav("wc_ts.wav", _sine(200, dur=0.5))
        results = process_file(path, _cfg(window_ms=25.0, hop_ms=10.0))
        hop_s   = 0.010
        try:
            for i in range(1, len(results)):
                diff = results[i].t_start_s - results[i-1].t_start_s
                self.assertAlmostEqual(diff, hop_s, places=4,
                    msg=f"Window {i}: step {diff:.5f}s ≠ {hop_s}s")
            _record("t_start increases by hop_ms each window", True)
        except AssertionError as e:
            _record("t_start increases by hop_ms each window", False, str(e)); raise

    def test_resampling_preserves_window_count(self):
        """A file recorded at 44.1 kHz resampled to 16 kHz gives ≈ same window
        count as the same signal natively recorded at 16 kHz (± 3 windows)."""
        # Generate audio at each target SR so both files have the same duration
        sig_44k = _sine(200, dur=1.0, sr=44100)
        sig_16k = _sine(200, dur=1.0, sr=16000)
        n_resampled = len(process_file(
            _write_wav("rs_44k.wav", sig_44k, sr=44100),
            _cfg(target_sr=16000)))
        n_native = len(process_file(
            _write_wav("rs_16k.wav", sig_16k, sr=16000),
            _cfg(target_sr=0)))
        try:
            self.assertAlmostEqual(n_resampled, n_native, delta=3,
                msg=f"Resampled 44k→16k: {n_resampled} windows, native 16k: {n_native}")
            _record("resampling preserves window count (±3)", True,
                    f"44k→16k={n_resampled}, native={n_native}")
        except AssertionError as e:
            _record("resampling preserves window count (±3)", False, str(e)); raise

    def test_empty_file_no_results(self):
        """Zero-length file → zero windows, no crash."""
        path = _write_wav("wc_empty.wav", np.array([], dtype=np.float32))
        try:
            results = process_file(path, _cfg())
            self.assertEqual(len(results), 0,
                f"Expected 0 windows for empty file, got {len(results)}")
            _record("empty file returns 0 windows", True)
        except Exception as e:
            _record("empty file returns 0 windows", False, str(e)); raise

    def test_all_numeric_fields_are_numeric_type(self):
        """Every numeric field in WindowResult is a float or numpy float (never None/str)."""
        results = process_file(_write_wav("wc_dtype.wav", _sine(200, dur=0.3)), _cfg())
        fields  = ["loudness_dbfs", "periodicity", "jitter_local", "shimmer_local",
                   "F1","B1","F2","B2","F3","B3","F4","B4","F5","B5","F6","B6","F7","B7"]
        try:
            for r in results:
                for f in fields:
                    v = getattr(r, f)
                    self.assertIsInstance(v, (float, np.floating),
                        f"Field {f} has type {type(v).__name__}, expected float")
            _record("all numeric fields are float/np.floating", True)
        except AssertionError as e:
            _record("all numeric fields are float/np.floating", False, str(e)); raise

    def test_window_index_sequential_from_zero(self):
        """window_index must be 0, 1, 2, ... for each file independently."""
        for name, sig in [("sine_idx", _sine(200)), ("noise_idx", _white_noise())]:
            results = process_file(_write_wav(f"{name}.wav", sig), _cfg())
            try:
                for expected_i, r in enumerate(results):
                    self.assertEqual(r.window_index, expected_i,
                        f"{name}: window {expected_i} has index {r.window_index}")
                _record(f"window_index sequential for {name}", True)
            except AssertionError as e:
                _record(f"window_index sequential for {name}", False, str(e)); raise


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary():
    if not _results:
        return
    passed = [r for r in _results if r["passed"]]
    failed = [r for r in _results if not r["passed"]]
    col_w  = max(len(r["test"]) for r in _results) + 2
    bar    = "=" * (col_w + 22)

    print(f"\n{bar}")
    print(f"  TEST SUMMARY  —  {len(passed)}/{len(_results)} passed")
    print(bar)
    for r in _results:
        icon   = "✓" if r["passed"] else "✗"
        detail = f"  [{r['detail']}]" if r["detail"] else ""
        print(f"  {icon}  {r['test']:<{col_w}}{detail}")
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for r in failed:
            print(f"     ✗  {r['test']}")
            if r["detail"]:
                print(f"        {r['detail']}")
    print(bar + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None    # preserve definition order
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    _print_summary()
    sys.exit(0 if result.wasSuccessful() else 1)
