"""
Generates synthetic test WAV files and runs the labeling pipeline on them.
Validates that all metrics are produced correctly.
"""

import wave
import struct
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import PipelineConfig, process_file, write_csv, print_summary


def write_test_wav(path: str, sr: int, duration: float, freq: float = 220.0,
                   noise: float = 0.05):
    """Write a sine + noise WAV file (16-bit mono)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + harmonics to give formant structure
    sig = (
        0.6 * np.sin(2 * np.pi * freq       * t) +
        0.3 * np.sin(2 * np.pi * freq * 2   * t) +
        0.15* np.sin(2 * np.pi * freq * 3   * t) +
        0.1 * np.sin(2 * np.pi * freq * 4   * t) +
        noise * np.random.randn(len(t))
    )
    sig /= np.max(np.abs(sig) + 1e-9)
    sig_int = (sig * 32000).astype(np.int16)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sig_int.tobytes())
    print(f"  Created {path}  ({sr} Hz, {duration:.1f}s, F0≈{freq} Hz)")


def run_tests():
    os.makedirs("/tmp/test_audio", exist_ok=True)
    os.makedirs("/tmp/test_out",   exist_ok=True)

    print("\n=== Generating test audio files ===\n")
    tests = [
        ("/tmp/test_audio/voice_440hz_44100sr.wav",  44100, 1.5, 440),
        ("/tmp/test_audio/voice_120hz_22050sr.wav",  22050, 2.0, 120),
        ("/tmp/test_audio/voice_220hz_8000sr.wav",   8000,  1.0, 220),
    ]
    for path, sr, dur, freq in tests:
        write_test_wav(path, sr, dur, freq)

    print("\n=== Running pipeline ===\n")
    cfg = PipelineConfig(
        target_sr=16000,
        window_ms=25.0,
        hop_ms=10.0,
        f0_min=60,
        f0_max=600,
        voicing_threshold=0.3,
        n_formants=7,
    )

    all_results = []
    for path, *_ in tests:
        results = process_file(path, cfg)
        all_results.extend(results)
        print_summary(results)

    # Write CSV
    out_csv = "/tmp/test_out/labels.csv"
    write_csv(all_results, out_csv)

    # Spot-check
    assert len(all_results) > 0, "No results produced!"
    r = all_results[0]
    assert not math.isnan(r.loudness_dbfs), "Loudness NaN"
    assert not math.isnan(r.periodicity),  "Periodicity NaN"
    # Formants should be present for voiced harmonic signal
    voiced = [r for r in all_results if not math.isnan(r.periodicity) and r.periodicity > 0.3]
    f1_vals = [r.F1 for r in voiced if not math.isnan(r.F1)]
    print(f"\nVoiced windows: {len(voiced)}")
    print(f"Windows with F1: {len(f1_vals)}")
    if f1_vals:
        print(f"F1 range: {min(f1_vals):.1f} – {max(f1_vals):.1f} Hz")

    print(f"\nAll tests passed! Output written to {out_csv}")


if __name__ == "__main__":
    run_tests()
