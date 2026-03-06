"""
Hillenbrand Dataset Runner
==========================
Downloads the Hillenbrand (1995) American English vowels dataset from HuggingFace,
saves each audio sample to a temporary WAV file, runs the labelling pipeline over
it, and writes the results to a CSV enriched with:

  - speaker  : parsed from the filename  (e.g. "m01ae.wav" → "m01")
  - vowel    : from the dataset's  vowel  field  (e.g. "ae")
  - groups   : from the dataset's  groups field  (e.g. "m" / "w" / "b" / "g")

Usage:
    python run_hillenbrand.py [--output results.csv] [--window 25] [--hop 10]
                              [--target-sr 16000] [--n-formants 7]
                              [--dataset-name <hf_repo_id>] [--split train]

The dataset identifier defaults to "speech-trove/hillenbrand" – change it with
--dataset-name if your copy lives elsewhere on the Hub.
"""

import argparse
import csv
import io
import logging
import re
import struct
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import the pipeline from the same directory (or adjust sys.path as needed)
# ---------------------------------------------------------------------------
from pipeline import PipelineConfig, process_file, FIELDNAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

# Hillenbrand filenames follow the pattern:  <group><speaker_num><vowel>.wav
# group:       one letter  – m (men), w (women), b (boys), g (girls)
# speaker_num: two digits  – 01 … 99
# vowel:       2–3 chars   – ae, ah, aw, eh, ei, er, ih, iy, oa, oo, uh, uw
_FNAME_RE = re.compile(
    r"^(?P<group>[mwbg])(?P<num>\d{2})(?P<vowel>[a-z]{2,3})(?:\.wav)?$",
    re.IGNORECASE,
)

def parse_filename(filename: str) -> dict:
    """
    Parse a Hillenbrand WAV filename into its components.

    Returns a dict with keys: 'speaker', 'vowel_parsed', 'group_parsed'.
    Falls back to empty strings on a parse failure so the row is never dropped.
    """
    stem = Path(filename).stem  # strip directory and extension
    m = _FNAME_RE.match(stem)
    if m:
        group  = m.group("group").lower()
        num    = m.group("num")
        vowel  = m.group("vowel").lower()
        return {
            "speaker":      f"{group}{num}",   # e.g. "m01"
            "vowel_parsed": vowel,              # e.g. "ae"
            "group_parsed": group,             # e.g. "m"
        }
    log.warning(f"Could not parse filename: {filename!r}")
    return {"speaker": "", "vowel_parsed": "", "group_parsed": ""}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def array_to_wav_bytes(array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy array (±1) to an in-memory 16-bit mono WAV."""
    pcm = np.clip(array, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)           # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    dataset_name: str,
    split: str,
    cfg: PipelineConfig,
    output_path: str,
):
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        raise SystemExit(
            "The `datasets` package is required.  Install it with:\n"
            "  pip install datasets soundfile"
        )

    log.info(f"Loading dataset: {dataset_name!r}  split={split!r}")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

    # Cast the audio column so HuggingFace decodes it for us
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=True))

    # Build the extended fieldnames: pipeline fields + our three extras
    extra_fields = ["speaker", "vowel", "groups"]
    fieldnames   = extra_fields + FIELDNAMES

    n_written = 0
    with open(output_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, sample in enumerate(ds):
            # ----------------------------------------------------------------
            # 1.  Extract metadata from the dataset row
            # ----------------------------------------------------------------
            audio_col  = sample.get("audio", {})
            filename   = audio_col.get("path", "") or sample.get("file_name", "")
            filename   = Path(filename).name  # keep just the basename

            # Dataset-level labels (fall back gracefully if absent)
            ds_vowel   = sample.get("vowel",  "")
            ds_groups  = sample.get("groups", "")

            # Parsed from filename
            parsed = parse_filename(filename)

            # Prefer the dataset's own vowel/groups fields; use parsed as backup
            vowel  = ds_vowel  or parsed["vowel_parsed"]
            groups = ds_groups or parsed["group_parsed"]
            speaker = parsed["speaker"]

            # ----------------------------------------------------------------
            # 2.  Write audio to a temporary WAV and run the pipeline
            # ----------------------------------------------------------------
            array       = audio_col.get("array")
            sample_rate = audio_col.get("sampling_rate", 16000)

            if array is None or len(array) == 0:
                log.warning(f"[{idx}] Empty audio for {filename!r}, skipping.")
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(array_to_wav_bytes(np.asarray(array, dtype=np.float32),
                                             sample_rate))

            try:
                window_results = process_file(tmp_path, cfg)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if not window_results:
                log.warning(f"[{idx}] No windows extracted for {filename!r}")
                continue

            # ----------------------------------------------------------------
            # 3.  Write one row per analysis window
            # ----------------------------------------------------------------
            for wr in window_results:
                row = wr.to_dict()
                # Overwrite the 'file' key with the original dataset filename
                row["file"]   = filename
                row["vowel"]  = vowel
                row["groups"] = groups
                row["speaker"] = speaker
                writer.writerow(row)
                n_written += 1

            if (idx + 1) % 50 == 0:
                log.info(f"  Processed {idx + 1} samples …")

    log.info(f"Done. {n_written} rows written to {output_path!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-label the Hillenbrand HuggingFace dataset with the "
                    "acoustic feature pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-name", default="speech-trove/hillenbrand",
                   help="HuggingFace dataset repository ID")
    p.add_argument("--split",  default="train",
                   help="Dataset split to process (e.g. 'train', 'all')")
    p.add_argument("--output", "-o", default="hillenbrand_labels.csv",
                   help="Output CSV path")

    # Pipeline parameters (mirrors pipeline.py CLI)
    p.add_argument("--target-sr",   type=int,   default=16000)
    p.add_argument("--window",      type=float, default=25.0,
                   help="Analysis window length (ms)")
    p.add_argument("--hop",         type=float, default=10.0,
                   help="Hop size (ms)")
    p.add_argument("--f0-min",      type=float, default=60.0)
    p.add_argument("--f0-max",      type=float, default=400.0)
    p.add_argument("--voicing-threshold", type=float, default=0.45)
    p.add_argument("--lpc-order",   type=int,   default=None,
                   help="LPC order (None = auto: 2 + sr/1000)")
    p.add_argument("--n-formants",  type=int,   default=7)
    return p


def main():
    args = build_parser().parse_args()

    cfg = PipelineConfig(
        target_sr         = args.target_sr,
        window_ms         = args.window,
        hop_ms            = args.hop,
        f0_min            = args.f0_min,
        f0_max            = args.f0_max,
        voicing_threshold = args.voicing_threshold,
        lpc_order         = args.lpc_order,
        n_formants        = min(7, max(1, args.n_formants)),
    )

    run(
        dataset_name = args.dataset_name,
        split        = args.split,
        cfg          = cfg,
        output_path  = args.output,
    )


if __name__ == "__main__":
    main()
