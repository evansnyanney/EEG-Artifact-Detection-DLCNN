#!/usr/bin/env python3
"""
EDF Channel Inspection Utility

Inspects a sample of EDF files to report channel names, counts, and sampling rates.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import os
import glob
from typing import List

import mne


def check_edf_channels(edf_glob: str, max_files: int = 5) -> None:
    """Inspect channel names and sampling rate for a subset of EDF files.

    Args:
        edf_glob: Glob pattern to locate EDF files.
        max_files: Maximum number of files to inspect.
    """
    edf_files: List[str] = sorted(glob.glob(edf_glob))
    print("EDF Channel Inspection")
    print("=" * 60)
    print(f"Search pattern: {edf_glob}")
    print(f"Found files: {len(edf_files)}")

    if not edf_files:
        print("No EDF files found.")
        return

    for i, edf_file in enumerate(edf_files[:max_files]):
        print(f"\nFile {i+1}: {os.path.basename(edf_file)}")
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            print(f"  Number of channels: {len(raw.ch_names)}")
            print(f"  Channel names: {raw.ch_names}")
            print(f"  Sampling rate: {raw.info['sfreq']} Hz")
            bipolar_like = [ch for ch in raw.ch_names if '-' in ch]
            print(f"  Channels with hyphens: {len(bipolar_like)}")
            if bipolar_like:
                print(f"  Example bipolar channels: {bipolar_like[:5]}")
        except Exception as e:
            print(f"  Error: {e}")
