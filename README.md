# Audio Sample Labeling Pipeline

A zero-dependency Python pipeline (requires only **numpy** and **scipy**) that
extracts acoustic features from WAV files over configurable time windows.

---

## Features extracted per window

| Column | Description |
|---|---|
| `loudness_dbfs` | RMS loudness in dBFS (0 dBFS = amplitude 1.0) |
| `periodicity` | Normalised autocorrelation peak in the F0 search range (0–1) |
| `jitter_local` | RAP jitter in % (voiced frames only) |
| `shimmer_local` | Local shimmer in % (voiced frames only) |
| `F1`–`F7` | Formant frequencies in Hz (LPC-based, as available) |
| `B1`–`B7` | Formant bandwidths in Hz (LPC-based, as available) |

Each row also includes: `file`, `window_index`, `t_center_s`, `t_start_s`, `t_end_s`.

---

## Requirements

```
numpy>=1.21
scipy>=1.7
```

No other libraries needed. The pipeline uses only Python's built-in `wave`
module for audio I/O, so it reads **mono or stereo 16-bit WAV files** natively.

---

## Quickstart

### Single file → CSV

```bash
python pipeline.py --input recording.wav --output labels.csv --summary
```

### Directory of files → JSON, custom windows

```bash
python pipeline.py \
  --input ./audio_dir/ \
  --output results.json \
  --target-sr 16000 \
  --window 40 \
  --hop 20
```

### Run the built-in tests

```bash
python test_pipeline.py
```

---

## CLI reference

```
usage: pipeline.py [-h] --input INPUT [--output OUTPUT]
                   [--target-sr TARGET_SR] [--window WINDOW] [--hop HOP]
                   [--f0-min F0_MIN] [--f0-max F0_MAX]
                   [--voicing-threshold VOICING_THRESHOLD]
                   [--lpc-order LPC_ORDER] [--n-formants N_FORMANTS]
                   [--summary]
```

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | WAV file **or** directory of WAV files |
| `--output` | `labels.csv` | Output path — `.csv` or `.json` |
| `--target-sr` | `16000` | Resample to this rate; `0` = keep original |
| `--window` | `25.0` | Analysis window length (ms) |
| `--hop` | `10.0` | Hop size between windows (ms) |
| `--f0-min` | `60` | Min F0 for pitch search (Hz) |
| `--f0-max` | `400` | Max F0 for pitch search (Hz) |
| `--voicing-threshold` | `0.45` | Autocorr threshold to count as voiced |
| `--lpc-order` | auto | LPC polynomial order (default: `2 + sr/1000`) |
| `--n-formants` | `7` | Max formant columns to emit (1–7) |
| `--summary` | off | Print per-file statistics to stdout |

---

## Using as a library

```python
from pipeline import PipelineConfig, process_file, write_csv

cfg = PipelineConfig(
    target_sr=16000,
    window_ms=25.0,
    hop_ms=10.0,
    f0_min=60,
    f0_max=400,
    voicing_threshold=0.45,
    n_formants=5,
)

results = process_file("my_recording.wav", cfg)
write_csv(results, "labels.csv")

# Or iterate manually
for r in results:
    print(r.t_center_s, r.loudness_dbfs, r.F1, r.F2)
```

---

## How each metric is computed

### Loudness
RMS amplitude of the window, expressed as 20·log₁₀(RMS) dBFS.
A full-scale sine wave gives ≈ −3 dBFS; silence gives −100 dBFS.

### Periodicity
Normalised autocorrelation at the dominant lag within `[sr/f0_max, sr/f0_min]`.
Values above `voicing_threshold` indicate a voiced frame.

### Jitter (RAP)
Relative Average Perturbation: the mean absolute deviation of each period
from the average of its two neighbours, divided by the mean period.
Only computed on voiced frames with at least 3 detected cycles.

### Shimmer (local)
Mean absolute amplitude difference between adjacent glottal cycles,
divided by the mean amplitude. Only computed on voiced frames.

### Formants F1–F7 / B1–B7
1. Apply pre-emphasis filter (α = 0.97)
2. Multiply by a Hamming window
3. Compute LPC coefficients via autocorrelation + Levinson-Durbin
4. Find roots of the LPC polynomial
5. Convert complex roots to frequency (Hz) and bandwidth (Hz)
6. Keep roots with positive imaginary part, 0 < f < Nyquist, bandwidth < 500 Hz
7. Sort ascending by frequency; emit up to `n_formants` pairs

---

## Input format

- **Format:** PCM WAV (`.wav`)
- **Bit depth:** 8, 16, or 32-bit integer
- **Channels:** Mono or stereo (stereo is mixed down to mono automatically)
- **Sample rate:** Any — resampled to `target_sr` if specified

---

## Output format

### CSV (default)
One row per window. NaN values are written as empty strings by default
in most spreadsheet applications.

### JSON (`--output results.json`)
A JSON array of objects, one per window, with NaN serialised as `null`
when loaded by standard parsers.
