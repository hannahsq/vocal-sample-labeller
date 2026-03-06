"""
Hillenbrand Dataset Runner
==========================
Downloads the MLSpeech/hillenbrand_vowels dataset from HuggingFace, runs the
acoustic labelling pipeline over each sample, and writes results to a CSV
enriched with:

  - speaker  : derived from 'group' field + per-group row index
               (e.g. "m01", "w03", "b02", "g07")
  - vowel    : from the dataset's 'vowel' field  (e.g. "ae")
  - groups   : from the dataset's 'group' field  (e.g. "m" / "w" / "b" / "g")

The dataset's audio column is a string-encoded Python list of float32 samples
at 16 kHz — no external audio decoder (torchcodec / soundfile) is needed.

Usage:
    python run_hillenbrand.py
    python run_hillenbrand.py --output results.csv --window 25 --hop 10
    python run_hillenbrand.py --split train --target-sr 16000 --n-formants 7
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

from pipeline import PipelineConfig, process_file, FIELDNAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# The MLSpeech/hillenbrand_vowels audio arrays are already at 16 kHz
DATASET_SR = 16000


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def array_to_wav_bytes(array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy array (±1) to in-memory 16-bit mono WAV bytes."""
    pcm = np.clip(array, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Speaker ID assignment
# ---------------------------------------------------------------------------

def make_speaker_assigner() -> callable:
    """
    Returns a callable get_speaker(group) that assigns a stable, incrementing
    two-digit index within each group as new samples are seen, producing IDs
    like 'm01', 'w03', 'b01', 'g10'.

    The Hillenbrand dataset has no explicit speaker column; each row is one
    isolated-vowel token recorded by a distinct speaker within a group.
    Incrementing per group faithfully reconstructs the original numbering
    (m01–m45, w01–w45, b01–b10, g01–g10) when the dataset is in its natural
    order.
    """
    counters: dict[str, int] = {}

    def get_speaker(group: str) -> str:
        counters[group] = counters.get(group, 0) + 1
        return f"{group}{counters[group]:02d}"

    return get_speaker


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    dataset_name: str,
    split: str,
    cfg: PipelineConfig,
    output_path: str,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The `datasets` package is required.\n"
            "Install it with:  pip install datasets"
        )

    log.info(f"Loading dataset: {dataset_name!r}  split={split!r}")
    ds = load_dataset(dataset_name, split=split, streaming=False)
    log.info(f"Dataset loaded — {len(ds)} samples")

    extra_fields = ["speaker", "vowel", "groups"]
    fieldnames   = extra_fields + FIELDNAMES

    get_speaker = make_speaker_assigner()

    n_written = 0
    with open(output_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, item in enumerate(ds):
            # ----------------------------------------------------------------
            # 1.  Decode the string-encoded audio array
            # ----------------------------------------------------------------
            try:
                audio = np.array(ast.literal_eval(item["audio"]), dtype=np.float32)
            except Exception as exc:
                log.warning(f"[{idx}] Could not decode audio: {exc} — skipping")
                continue

            if audio.size == 0:
                log.warning(f"[{idx}] Empty audio — skipping")
                continue

            # ----------------------------------------------------------------
            # 2.  Extract metadata
            #     Note: the dataset uses 'group' (singular), not 'groups'
            # ----------------------------------------------------------------
            vowel  = item.get("vowel", "")
            groups = item.get("group", "")
            speaker = get_speaker(groups)

            # Synthetic filename used for the pipeline's 'file' column
            filename = f"{speaker}{vowel}.wav"

            # ----------------------------------------------------------------
            # 3.  Write audio to a temp WAV, run the pipeline, clean up
            # ----------------------------------------------------------------
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(array_to_wav_bytes(audio, DATASET_SR))

            try:
                window_results = process_file(tmp_path, cfg)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if not window_results:
                log.warning(f"[{idx}] No windows extracted for {filename!r}")
                continue

            # ----------------------------------------------------------------
            # 4.  Write one CSV row per analysis window
            # ----------------------------------------------------------------
            for wr in window_results:
                row = wr.to_dict()
                row["file"]    = filename
                row["vowel"]   = vowel
                row["groups"]  = groups
                row["speaker"] = speaker
                writer.writerow(row)
                n_written += 1

            if (idx + 1) % 100 == 0:
                log.info(f"  Processed {idx + 1} / {len(ds)} samples …")

    log.info(f"Done — {n_written} rows written to {output_path!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-label MLSpeech/hillenbrand_vowels with the acoustic "
                    "feature pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-name", default="MLSpeech/hillenbrand_vowels",
                   help="HuggingFace dataset repository ID")
    p.add_argument("--split",  default="train",
                   help="Dataset split to process")
    p.add_argument("--output", "-o", default="hillenbrand_labels.csv",
                   help="Output CSV path")

    # Pipeline parameters
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


def main() -> None:
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
