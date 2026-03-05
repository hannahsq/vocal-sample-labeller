"""
Audio Sample Labeling Pipeline
================================
Extracts acoustic features from audio files over configurable time windows.

Metrics extracted per window:
  - Loudness (dBFS relative to amplitude 1.0)
  - Periodicity (autocorrelation-based)
  - Shimmer (amplitude perturbation)
  - Jitter (fundamental frequency perturbation)
  - Formant frequencies F1–F7 and bandwidths B1–B7 (LPC-based)

Usage:
    python pipeline.py --input path/to/audio.wav --window 25 --hop 10
    python pipeline.py --input dir/ --target-sr 16000 --window 40 --hop 20 --output results.csv
"""

import argparse
import csv
import json
import logging
import os
import struct
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy.signal import lfilter, resample_poly
from scipy.linalg import solve_toeplitz
from math import gcd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tunable parameters for the pipeline."""
    # Resampling
    target_sr: int = 16000          # Target sample rate (Hz); 0 = keep original

    # Windowing
    window_ms: float = 25.0         # Analysis window length (ms)
    hop_ms: float = 10.0            # Hop size between windows (ms)

    # Pitch / periodicity
    f0_min: float = 60.0            # Min F0 for pitch search (Hz)
    f0_max: float = 400.0           # Max F0 for pitch search (Hz)
    voicing_threshold: float = 0.45 # Min normalised autocorr to be voiced

    # LPC / formants
    lpc_order: Optional[int] = None # None → auto: 2 + sr/1000
    n_formants: int = 7             # How many formants to extract (up to 7)

    # Jitter / shimmer
    max_period_ratio: float = 1.3   # Max ratio of consecutive periods (RAP guard)


@dataclass
class WindowResult:
    """Feature values for a single analysis window."""
    file: str = ""
    window_index: int = 0
    t_center_s: float = 0.0
    t_start_s: float = 0.0
    t_end_s: float = 0.0

    # Core features
    loudness_dbfs: float = float("nan")
    periodicity: float = float("nan")
    jitter_local: float = float("nan")   # RAP jitter (%)
    shimmer_local: float = float("nan")  # Local shimmer (%)

    # Formants / bandwidths (F1-F7, B1-B7)
    F1: float = float("nan"); B1: float = float("nan")
    F2: float = float("nan"); B2: float = float("nan")
    F3: float = float("nan"); B3: float = float("nan")
    F4: float = float("nan"); B4: float = float("nan")
    F5: float = float("nan"); B5: float = float("nan")
    F6: float = float("nan"); B6: float = float("nan")
    F7: float = float("nan"); B7: float = float("nan")

    def set_formant(self, idx: int, freq: float, bw: float):
        """Set F{idx} and B{idx} (idx 1-based)."""
        setattr(self, f"F{idx}", freq)
        setattr(self, f"B{idx}", bw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file. Returns (float32 mono array normalised to ±1, sample_rate)."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        sr         = wf.getframerate()
        n_frames   = wf.getnframes()
        raw        = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    samples = np.frombuffer(raw, dtype=dtype_map[sampwidth]).astype(np.float32)
    samples = samples.reshape(-1, n_channels).mean(axis=1)          # mono mix
    samples /= float(2 ** (8 * sampwidth - 1))                      # → ±1
    samples = np.clip(samples, -1.0, 1.0)
    return samples, sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using polyphase method."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def loudness_dbfs(frame: np.ndarray) -> float:
    """RMS loudness in dBFS (0 dBFS = amplitude 1.0)."""
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)


# --- Autocorrelation pitch & periodicity ---

def _autocorr(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation via FFT."""
    n = len(x)
    X = np.fft.rfft(x, n=2 * n)
    r = np.fft.irfft(X * np.conj(X))[:n].real
    r0 = r[0]
    return r / r0 if r0 > 1e-12 else r


def pitch_and_periodicity(
    frame: np.ndarray,
    sr: int,
    f0_min: float,
    f0_max: float,
    voicing_threshold: float
) -> Tuple[Optional[float], float]:
    """
    Returns (f0_hz_or_None, periodicity_0_to_1).
    periodicity is the normalised autocorrelation peak in the valid F0 range.
    """
    ac = _autocorr(frame)
    lag_min = int(sr / f0_max)
    lag_max = int(sr / f0_min)
    lag_max = min(lag_max, len(ac) - 1)

    if lag_min >= lag_max:
        return None, 0.0

    segment = ac[lag_min:lag_max + 1]
    peak_idx = int(np.argmax(segment))
    periodicity = float(segment[peak_idx])
    lag = lag_min + peak_idx
    f0 = sr / lag if lag > 0 else None

    if periodicity < voicing_threshold:
        return None, periodicity
    return f0, periodicity


# --- Jitter & shimmer (require voiced frames with detected periods) ---

def _extract_periods(frame: np.ndarray, sr: int, f0: float) -> List[float]:
    """
    Extract individual cycle lengths (in samples) using zero-crossing intervals
    guided by estimated F0 period.
    Returns list of period lengths (in seconds).
    """
    expected_period = sr / f0
    # Find zero crossings (positive-going)
    zc = np.where((frame[:-1] < 0) & (frame[1:] >= 0))[0]
    if len(zc) < 4:
        return []

    periods = np.diff(zc.astype(float)) / sr  # seconds
    # Filter implausible periods
    med = np.median(periods)
    periods = periods[(periods > 0.4 * med) & (periods < 2.5 * med)]
    return periods.tolist()


def _peak_amplitudes(frame: np.ndarray, sr: int, f0: float) -> List[float]:
    """Amplitude of each glottal cycle using a simple peak-picking approach."""
    win = int(round(sr / f0))
    if win < 2:
        return []
    amps = []
    i = 0
    while i + win <= len(frame):
        segment = np.abs(frame[i:i + win])
        amps.append(float(np.max(segment)))
        i += win
    return amps


def jitter_rap(periods: List[float]) -> float:
    """
    Relative Average Perturbation (RAP) jitter in percent.
    Average absolute difference of each period from the mean of its neighbours.
    """
    if len(periods) < 3:
        return float("nan")
    p = np.array(periods)
    rap_num = np.mean(np.abs(p[1:-1] - (p[:-2] + p[1:-1] + p[2:]) / 3.0))
    return 100.0 * rap_num / np.mean(p)


def shimmer_local(amps: List[float]) -> float:
    """
    Local shimmer in percent.
    Mean absolute amplitude difference between consecutive cycles / mean amplitude.
    """
    if len(amps) < 2:
        return float("nan")
    a = np.array(amps)
    return 100.0 * np.mean(np.abs(np.diff(a))) / np.mean(a)


# --- LPC-based formant extraction ---

def lpc_coeff(frame: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients using Levinson-Durbin recursion."""
    # Autocorrelation method
    r = np.correlate(frame, frame, mode="full")
    r = r[len(frame) - 1:]          # one-sided
    r = r[:order + 1]

    if np.abs(r[0]) < 1e-12:
        return np.zeros(order)

    try:
        a = solve_toeplitz(r[:order], -r[1:order + 1])
    except np.linalg.LinAlgError:
        return np.zeros(order)

    return np.concatenate([[1.0], a])


def formants_from_lpc(
    frame: np.ndarray,
    sr: int,
    order: int,
    n_formants: int
) -> List[Tuple[float, float]]:
    """
    Extract formant frequencies and bandwidths from LPC roots.
    Returns list of (freq_hz, bandwidth_hz) sorted by frequency,
    filtered to 0–(sr/2) Hz and bandwidth < 500 Hz.
    """
    # Pre-emphasis
    pre = lfilter([1.0, -0.97], [1.0], frame)
    # Hamming window
    pre *= np.hamming(len(pre))

    a = lpc_coeff(pre, order)
    if np.all(a == 0):
        return []

    roots = np.roots(a)
    # Keep roots with positive imaginary part (complex pairs)
    roots = roots[np.imag(roots) >= 0]

    formants = []
    for root in roots:
        angle = np.angle(root)
        freq = angle * (sr / (2.0 * np.pi))
        if freq <= 0 or freq >= sr / 2:
            continue
        # Bandwidth from root magnitude
        bw = -np.log(np.abs(root)) * (sr / np.pi)
        if bw <= 0 or bw > 500:
            continue
        formants.append((freq, bw))

    formants.sort(key=lambda x: x[0])
    return formants[:n_formants]


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def analyse_frame(
    frame: np.ndarray,
    sr: int,
    cfg: PipelineConfig,
    file_name: str,
    win_idx: int,
    t_start: float,
    t_end: float,
) -> WindowResult:
    res = WindowResult(
        file=file_name,
        window_index=win_idx,
        t_center_s=(t_start + t_end) / 2,
        t_start_s=t_start,
        t_end_s=t_end,
    )

    if len(frame) == 0:
        return res

    # 1. Loudness
    res.loudness_dbfs = loudness_dbfs(frame)

    # 2. Periodicity + F0
    f0, periodicity = pitch_and_periodicity(
        frame, sr, cfg.f0_min, cfg.f0_max, cfg.voicing_threshold
    )
    res.periodicity = periodicity

    # 3. Jitter & shimmer (voiced frames only)
    if f0 is not None:
        periods = _extract_periods(frame, sr, f0)
        amps    = _peak_amplitudes(frame, sr, f0)
        res.jitter_local  = jitter_rap(periods)
        res.shimmer_local = shimmer_local(amps)

    # 4. Formants
    lpc_order = cfg.lpc_order or max(8, 2 + int(sr / 1000))
    fmts = formants_from_lpc(frame, sr, lpc_order, cfg.n_formants)
    for i, (freq, bw) in enumerate(fmts, start=1):
        res.set_formant(i, round(freq, 2), round(bw, 2))

    return res


# ---------------------------------------------------------------------------
# File-level pipeline
# ---------------------------------------------------------------------------

def process_file(
    path: str,
    cfg: PipelineConfig,
) -> List[WindowResult]:
    """Load, resample if needed, window, and extract features from one file."""
    log.info(f"Processing: {path}")

    try:
        audio, sr = read_wav(path)
    except Exception as e:
        log.error(f"Could not read {path}: {e}")
        return []

    # Resample
    if cfg.target_sr > 0 and sr != cfg.target_sr:
        log.info(f"  Resampling {sr} → {cfg.target_sr} Hz")
        audio = resample(audio, sr, cfg.target_sr)
        sr = cfg.target_sr

    win_samples = int(round(cfg.window_ms * sr / 1000.0))
    hop_samples = int(round(cfg.hop_ms   * sr / 1000.0))

    if win_samples < 32:
        log.warning("Window too small — increase window_ms")
        return []

    results = []
    n = len(audio)
    win_idx = 0
    start = 0

    while start + win_samples <= n:
        end   = start + win_samples
        frame = audio[start:end].copy()

        t_start = start / sr
        t_end   = end   / sr

        res = analyse_frame(
            frame, sr, cfg,
            file_name=os.path.basename(path),
            win_idx=win_idx,
            t_start=t_start,
            t_end=t_end,
        )
        results.append(res)
        start  += hop_samples
        win_idx += 1

    log.info(f"  → {len(results)} windows extracted")
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "file", "window_index", "t_center_s", "t_start_s", "t_end_s",
    "loudness_dbfs", "periodicity", "jitter_local", "shimmer_local",
    "F1","B1","F2","B2","F3","B3","F4","B4","F5","B5","F6","B6","F7","B7",
]


def write_csv(results: List[WindowResult], output_path: str):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    log.info(f"Saved CSV → {output_path}")


def write_json(results: List[WindowResult], output_path: str):
    data = [r.to_dict() for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, allow_nan=True)
    log.info(f"Saved JSON → {output_path}")


def print_summary(results: List[WindowResult]):
    if not results:
        print("No results.")
        return
    loudness  = [r.loudness_dbfs  for r in results if not np.isnan(r.loudness_dbfs)]
    periodic  = [r.periodicity    for r in results if not np.isnan(r.periodicity)]
    jitter    = [r.jitter_local   for r in results if not np.isnan(r.jitter_local)]
    shimmer   = [r.shimmer_local  for r in results if not np.isnan(r.shimmer_local)]

    print(f"\n{'='*55}")
    print(f" Summary  ({len(results)} windows from {results[0].file})")
    print(f"{'='*55}")
    def stat(name, vals, unit=""):
        if vals:
            print(f"  {name:<22}  mean={np.mean(vals):8.3f}{unit}  "
                  f"min={np.min(vals):8.3f}{unit}  max={np.max(vals):8.3f}{unit}")
        else:
            print(f"  {name:<22}  (no data)")
    stat("Loudness",    loudness, " dBFS")
    stat("Periodicity", periodic)
    stat("Jitter (RAP)", jitter,   " %")
    stat("Shimmer",      shimmer,  " %")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_wav_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        return sorted(str(f) for f in p.rglob("*.wav"))
    else:
        raise FileNotFoundError(f"Not found: {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audio feature labeling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,
                   help="WAV file or directory of WAV files")
    p.add_argument("--output", "-o", default="labels.csv",
                   help="Output file (.csv or .json)")
    p.add_argument("--target-sr", type=int, default=16000,
                   help="Target sample rate (0 = keep original)")
    p.add_argument("--window",  type=float, default=25.0,
                   help="Analysis window length (ms)")
    p.add_argument("--hop",     type=float, default=10.0,
                   help="Hop size between windows (ms)")
    p.add_argument("--f0-min",  type=float, default=60.0)
    p.add_argument("--f0-max",  type=float, default=400.0)
    p.add_argument("--voicing-threshold", type=float, default=0.45)
    p.add_argument("--lpc-order", type=int, default=None,
                   help="LPC order (default: auto = 2 + sr/1000)")
    p.add_argument("--n-formants", type=int, default=7,
                   help="Max formants to extract (1–7)")
    p.add_argument("--summary", action="store_true",
                   help="Print summary statistics to stdout")
    return p


def main():
    args = build_parser().parse_args()

    cfg = PipelineConfig(
        target_sr          = args.target_sr,
        window_ms          = args.window,
        hop_ms             = args.hop,
        f0_min             = args.f0_min,
        f0_max             = args.f0_max,
        voicing_threshold  = args.voicing_threshold,
        lpc_order          = args.lpc_order,
        n_formants         = min(7, max(1, args.n_formants)),
    )

    files = collect_wav_files(args.input)
    if not files:
        log.error("No WAV files found.")
        return

    all_results: List[WindowResult] = []
    for f in files:
        all_results.extend(process_file(f, cfg))

    if not all_results:
        log.warning("No features extracted.")
        return

    if args.summary:
        print_summary(all_results)

    out = args.output
    if out.endswith(".json"):
        write_json(all_results, out)
    else:
        write_csv(all_results, out)


if __name__ == "__main__":
    main()
