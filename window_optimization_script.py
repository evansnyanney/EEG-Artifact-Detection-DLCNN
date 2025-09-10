#!/usr/bin/env python3
"""
Window Size Optimization Script

Runs the end-to-end pipeline for several window sizes using non-overlapping
windows, without changing the core training/evaluation code. Summaries are
reported per window to support method selection and reporting.
"""

import os
import sys
import json
import subprocess
import argparse


def _python_exe() -> str:
    # Use current interpreter to avoid env mismatches
    return sys.executable or "python"


def _run(cmd, cwd=None, timeout=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding='utf-8',
        errors='ignore',
        env=env,
    )


def _extract_metric_any(text: str, labels) -> float:
    if isinstance(labels, str):
        labels = [labels]
    try:
        for line in text.splitlines():
            for label in labels:
                if label in line:
                    try:
                        value = line.split(":", 1)[1].strip()
                        value = value.split("(")[0].strip()
                        # remove unicode and extra symbols
                        value = value.replace("≤", "").replace("%", "")
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
    # Normalize keys to float window sizes and cast metrics to float
    def _sf(x):
        try:
            return float(x)
        except Exception:
            return 0.0
    ok = {}
    for k, v in results.items():
        if isinstance(v, dict) and v.get('success'):
            try:
                w = float(k)
            except Exception:
                continue
            ok[w] = v
    if not ok:
        print("No successful runs to summarize.")
        return
    print("Window | Accuracy | Precision | Recall | Specificity | ROC AUC | pROC@0.1 | F1")
    print("-" * 100)
    for w in sorted(ok.keys()):
        r = ok[w]
        print(f"{w:6.1f}s |   {_sf(r.get('accuracy')):.3f}   |   {_sf(r.get('precision')):.3f}   |  {_sf(r.get('sensitivity')):.3f}   |    {_sf(r.get('specificity')):.3f}   | {_sf(r.get('roc_auc')):.3f}   |   {_sf(r.get('partial_roc_auc')):.4f}   | {_sf(r.get('f1_score')):.3f}")
    best_sens = max(ok.items(), key=lambda x: _sf(x[1].get('sensitivity')))
    best_roc = max(ok.items(), key=lambda x: _sf(x[1].get('roc_auc')))
    best_proc = max(ok.items(), key=lambda x: _sf(x[1].get('partial_roc_auc')))
    print("\nBest sensitivity:", f"{_sf(best_sens[1].get('sensitivity')):.3f} at {best_sens[0]}s")
    print("Best ROC AUC:", f"{_sf(best_roc[1].get('roc_auc')):.3f} at {best_roc[0]}s")
    print("Best pROC@0.1:", f"{_sf(best_proc[1].get('partial_roc_auc')):.4f} at {best_proc[0]}s")


def _sweep(target_name: str, detector_script: str, results_file: str, args):
    pretty = {
        'eye_movement': 'EYE MOVEMENT',
        'muscle_artifacts': 'MUSCLE ARTIFACT',
        'non_physiological': 'NON-PHYSIOLOGICAL'
    }.get(target_name, target_name.upper())
    print(f"Window sweep: {pretty}")

    window_sizes = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    # Resume if previous results exist
    results_path = results_file

    if args.clear and os.path.exists(results_path):
        try:
            os.remove(results_path)
            print('Cleared previous results file.')
        except Exception:
            pass

    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            results = {}
    else:
        results = {}

    for w in window_sizes:
        print(f"\nWindow {w:.1f}s")
        try:
            # Skip windows already completed successfully
            if (not args.force) and str(w) in results and results[str(w)].get('success'):
                print("Already done; skipping.")
                continue

            # 1) Preprocess
            print("Preprocessing...")
            r1 = _run([_python_exe(), "eeg_preprocessing_pipeline.py", "--window-seconds", str(w), "--overlap", "0.0"], timeout=3600)
            if r1.returncode != 0:
                print("Preprocessing failed.")
                results[str(w)] = {"success": False, "error": "preprocessing"}
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                continue

            # 2) Binary prep
            print("Preparing binary data...")
            r2 = _run([_python_exe(), "binary_model_data_preparation.py"], timeout=1200)
            if r2.returncode != 0:
                print("Binary preparation failed.")
                results[str(w)] = {"success": False, "error": "binary_prep"}
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                continue

            # 3) Train detector
            print("Training...")
            try:
                r3 = _run([_python_exe(), detector_script], timeout=2400)
            except KeyboardInterrupt:
                print("Training interrupted; continuing.")
                results[str(w)] = {"success": False, "error": "interrupted"}
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                continue
            if r3.returncode != 0:
                print("Training failed.")
                results[str(w)] = {"success": False, "error": "training"}
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                continue

            out = r3.stdout
            metrics = {
                'accuracy': _extract_metric_any(out, ['Accuracy', 'Accuracy     :']),
                'precision': _extract_metric_any(out, ['Precision', 'Precision    :']),
                'roc_auc': _extract_metric_any(out, ['AUC (ROC)', 'ROC AUC']),
                'pr_auc': _extract_metric_any(out, ['AUC (PR)', 'PR AUC (original)']),
                'pr_auc_adj': _extract_metric_any(out, ['PR-AUC (Prevalence-Adjusted', 'PR AUC (prevalence-adj)']),
                'partial_roc_auc': _extract_metric_any(out, ['Partial ROC AUC (FPR<=0.1)', 'Partial ROC AUC (FPR≤0.1)']),
                'sensitivity': _extract_metric_any(out, ['Sensitivity', 'Recall       :', 'Recall     :']),
                'specificity': _extract_metric_any(out, ['Specificity']),
                'f1_score': _extract_metric_any(out, ['F1-Score', 'F1 Score'])
            }
            metrics['success'] = True
            results[str(w)] = metrics
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"sens={metrics['sensitivity']:.3f}, spec={metrics['specificity']:.3f}, roc={metrics['roc_auc']:.3f}, pROC@0.1={metrics['partial_roc_auc']:.4f}")

        except subprocess.TimeoutExpired:
            print("Timeout.")
            results[str(w)] = {"success": False, "error": "timeout"}
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error: {e}")
            results[str(w)] = {"success": False, "error": str(e)}
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    _print_summary(results, f"{pretty}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='rerun even if a window size is already completed')
    parser.add_argument('--clear', action='store_true', help='delete previous results before running')
    parser.add_argument('--target', type=str, default='eye', choices=['eye','muscle','nonphys','all'], help='which model to sweep')
    args = parser.parse_args()

    if args.target in ('eye','all'):
        _sweep('eye_movement', 'eye_movement_detector.py', 'eye_movement_window_optimization_results.json', args)
    if args.target in ('muscle','all'):
        _sweep('muscle_artifacts', 'muscle_artifact_detector.py', 'muscle_window_optimization_results.json', args)
    if args.target in ('nonphys','all'):
        _sweep('non_physiological', 'non_physiological_detector.py', 'nonphys_window_optimization_results.json', args)


