#!/usr/bin/env python3
"""
EDF File Property Inspection

This utility inspects a subset of EDF files to report sampling rates,
channel counts, duration, and estimated sample counts after resampling
to a target frequency. It helps diagnose window length mismatches that
can occur after resampling.
"""

import os
import glob
import argparse
from typing import List

import mne

def inspect_edf_properties(edf_glob: str, max_files: int, target_rate: int = 250) -> None:
    """Inspect EDF properties and estimate resampled sample counts.

    Args:
        edf_glob: Glob pattern to locate EDF files.
        max_files: Maximum number of files to inspect.
        target_rate: Target sampling rate for estimation (default 250 Hz).
    """

    files: List[str] = sorted(glob.glob(edf_glob))[:max_files]
    print("EDF File Property Inspection")
    print("=" * 60)
    print(f"Search pattern: {edf_glob}")
    print(f"Found files: {len(files)}")

    for i, edf_file in enumerate(files):
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            filename = os.path.basename(edf_file)
            print(f"{i+1}. {filename}")
            print(f"   Sampling rate: {raw.info['sfreq']} Hz")
            print(f"   Channels: {len(raw.ch_names)}")
            print(f"   Duration: {raw.times[-1]:.1f} seconds")
            print(f"   Total samples: {len(raw.times)}")

            fs = float(raw.info['sfreq'])
            if fs != float(target_rate):
                resampled_samples = int(len(raw.times) * float(target_rate) / fs)
                print(f"   After resampling to {target_rate} Hz: {resampled_samples} samples")
            else:
                print(f"   Already at target rate; no resampling needed")
            print()

        except Exception as e:
            print(f"Error reading {edf_file}: {e}")
            print()

    print("Summary note:")
    print("Window boundary warnings can occur if original sampling rates differ and data are resampled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect EDF sampling rates, channels, and durations.")
    parser.add_argument("--pattern", type=str, default="edf/01_tcp_ar/*.edf", help="Glob pattern for EDF files")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of files to inspect")
    parser.add_argument("--target-rate", type=int, default=250, help="Target sampling rate for estimation")
    args = parser.parse_args()

    inspect_edf_properties(args.pattern, args.max_files, args.target_rate)