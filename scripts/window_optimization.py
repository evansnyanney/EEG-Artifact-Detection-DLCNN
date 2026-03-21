#!/usr/bin/env python3
"""
Window Size Optimization Script

Runs the end-to-end pipeline for several window sizes using non-overlapping
windows. Summaries are reported per window to support method selection.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import os
import sys
import json
import subprocess
import argparse


def _python_exe() -> str:
    return sys.executable or "python"


def _run(cmd, cwd=None, timeout=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, encoding='utf-8', errors='ignore', env=env,
    )


def _extract_metric(text: str, labels) -> float:
    if isinstance(labels, str):
        labels = [labels]
    try:
        for line in text.splitlines():
            for label in labels:
                if label in line:
                    try:
                        value = line.split(":", 1)[1].strip().split("(")[0].strip()
                        value = value.replace("<=", "").replace("%", "")
                        return float(value)
                    except Exception:
                        continue
    except Exception:
        pass
    return 0.0


def _print_summary(results: dict, title: str) -> None:
    print("\n" + "=" * 80)
    print(f"{title} WINDOW OPTIMIZATION SUMMARY")
    print("=" * 80)

    ok = {}
    for k, v in results.items():
        if isinstance(v, dict) and v.get('success'):
            try:
                ok[float(k)] = v
            except Exception:
                continue

    if not ok:
        print("No successful runs to summarize.")
        return

    def sf(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    print("Window | Accuracy | Precision | Recall | Specificity | ROC AUC | pROC@0.1 | F1")
    print("-" * 100)
    for w in sorted(ok.keys()):
        r = ok[w]
        print(
            f"{w:6.1f}s | {sf(r.get('accuracy')):7.3f}   | {sf(r.get('precision')):9.3f}   | "
            f"{sf(r.get('sensitivity')):6.3f}   | {sf(r.get('specificity')):11.3f}   | "
            f"{sf(r.get('roc_auc')):7.3f}   | {sf(r.get('partial_roc_auc')):8.4f}   | "
            f"{sf(r.get('f1_score')):5.3f}"
        )


def _sweep(target_name, detector_script, results_file, args):
    pretty = {
        'eye_movement': 'EYE MOVEMENT',
        'muscle_artifacts': 'MUSCLE ARTIFACT',
        'non_physiological': 'NON-PHYSIOLOGICAL'
    }.get(target_name, target_name.upper())

    print(f"Window sweep: {pretty}")
    window_sizes = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0]

    if args.clear and os.path.exists(results_file):
        os.remove(results_file)

    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            results = {}

    for w in window_sizes:
        print(f"\nWindow {w:.1f}s")
        try:
            if not args.force and str(w) in results and results[str(w)].get('success'):
                print("Already done; skipping.")
                continue

            print("Preprocessing...")
            r1 = _run([_python_exe(), "-m", "scripts.preprocess",
                        "--window-seconds", str(w), "--overlap", "0.0"], timeout=3600)
            if r1.returncode != 0:
                results[str(w)] = {"success": False, "error": "preprocessing"}
                continue

            print("Preparing binary data...")
            r2 = _run([_python_exe(), "-m", "scripts.prepare_data"], timeout=1200)
            if r2.returncode != 0:
                results[str(w)] = {"success": False, "error": "binary_prep"}
                continue

            print("Training...")
            r3 = _run([_python_exe(), "-m", f"scripts.{detector_script}"], timeout=2400)
            if r3.returncode != 0:
                results[str(w)] = {"success": False, "error": "training"}
                continue

            out = r3.stdout
            metrics = {
                'accuracy': _extract_metric(out, ['Accuracy']),
                'precision': _extract_metric(out, ['Precision']),
                'roc_auc': _extract_metric(out, ['AUC (ROC)']),
                'partial_roc_auc': _extract_metric(out, ['Partial ROC AUC (FPR<=0.1)']),
                'sensitivity': _extract_metric(out, ['Sensitivity', 'Recall']),
                'specificity': _extract_metric(out, ['Specificity']),
                'f1_score': _extract_metric(out, ['F1-Score']),
                'success': True,
            }
            results[str(w)] = metrics

        except subprocess.TimeoutExpired:
            results[str(w)] = {"success": False, "error": "timeout"}
        except Exception as e:
            results[str(w)] = {"success": False, "error": str(e)}
        finally:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

    _print_summary(results, pretty)


def main():
    parser = argparse.ArgumentParser(description="Window size optimization sweep")
    parser.add_argument('--force', action='store_true', help='Rerun completed windows')
    parser.add_argument('--clear', action='store_true', help='Clear previous results')
    parser.add_argument('--target', type=str, default='eye',
                        choices=['eye', 'muscle', 'nonphys', 'all'])
    args = parser.parse_args()

    if args.target in ('eye', 'all'):
        _sweep('eye_movement', 'train_eye', 'eye_movement_window_results.json', args)
    if args.target in ('muscle', 'all'):
        _sweep('muscle_artifacts', 'train_muscle', 'muscle_window_results.json', args)
    if args.target in ('nonphys', 'all'):
        _sweep('non_physiological', 'train_nonphys', 'nonphys_window_results.json', args)


if __name__ == "__main__":
    main()
