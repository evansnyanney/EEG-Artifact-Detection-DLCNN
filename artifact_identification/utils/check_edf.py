#!/usr/bin/env python3
"""
EDF File Property Inspection

Inspects a subset of EDF files to report sampling rates, channel counts,
duration, and estimated sample counts after resampling.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import os
import glob
from typing import List

import mne


def inspect_edf_properties(
    edf_glob: str, max_files: int = 10, target_rate: int = 250
) -> None:
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
                resampled = int(len(raw.times) * float(target_rate) / fs)
                print(f"   After resampling to {target_rate} Hz: {resampled} samples")
            else:
                print(f"   Already at target rate")
            print()
        except Exception as e:
            print(f"Error reading {edf_file}: {e}\n")
