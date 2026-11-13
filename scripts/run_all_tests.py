"""FloodGuard AI - Automated Test Runner & Verification Suite.

Executes static syntax checks, unit tests, integration tests, end-to-end ML
pipeline verification, and generates structured test reports (JSON, Markdown,
and JUnit XML).
"""

import sys
import os
import time
import json
import platform
import compileall
import subprocess
import numpy as np
from datetime import datetime

# Set up project root in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_syntax_checks():
    """Verify syntax of all Python files in the repository."""
    print("==================================================================")
    print("STEP 1: STATIC & SYNTAX COMPILATION AUDIT")
    print("==================================================================")
    start = time.time()
    success = compileall.compile_dir(PROJECT_ROOT, maxlevels=10, quiet=1)
    duration = time.time() - start
    status = "PASSED" if success else "FAILED"
    print(f"Syntax audit status: {status} (checked in {duration:.2f}s)")
    return {
        "check": "static_syntax_compilation",
        "status": status,
        "duration_seconds": round(duration, 3),
        "details": "compileall verified all .py files in project tree.",
    }


def run_pytest_suite():
    """Execute pytest with JUnit XML output."""
    print("\n==================================================================")
    print("STEP 2: UNIT & INTEGRATION TEST EXECUTION (PYTEST)")
    print("==================================================================")
    junit_xml = os.path.join(REPORTS_DIR, "junit_report.xml")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        os.path.join(PROJECT_ROOT, "tests"),
        "-v",
        f"--junitxml={junit_xml}",
    ]
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    duration = time.time() - start
    print(proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr)

    passed_tests = []
    failed_tests = []
    skipped_tests = []

    for line in proc.stdout.splitlines():
        if " PASSED" in line:
            test_name = line.split()[0]
            passed_tests.append(test_name)
        elif " FAILED" in line:
            test_name = line.split()[0]
            failed_tests.append(test_name)
        elif " SKIPPED" in line:
            test_name = line.split()[0]
            skipped_tests.append(test_name)

    status = "PASSED" if proc.returncode == 0 else "FAILED"
    return {
        "check": "pytest_automated_suite",
        "status": status,
        "return_code": proc.returncode,
        "duration_seconds": round(duration, 3),
        "passed_count": len(passed_tests),
        "failed_count": len(failed_tests),
        "skipped_count": len(skipped_tests),
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "skipped_tests": skipped_tests,
    }


def run_ml_pipeline_validation():
    """Execute the end-to-end Machine Learning and Geospatial pipeline."""
    print("\n==================================================================")
    print("STEP 3: MACHINE LEARNING & GEOSPATIAL PIPELINE VERIFICATION")
    print("==================================================================")
    start = time.time()
    metrics = {}

    from config import DATA_DIR, FEATURES, TARGET_REGRESSION, TEST_SPLIT_RATIO, RISK_CATEGORIES
    from src.data_preprocessing import load_data, clean_data, engineer_features, chronological_split, normalize_features, create_sequences
    from src.lstm_model import build_lstm_model, train_lstm_model, predict_lstm, TF_AVAILABLE
    from src.ml_benchmarking import compare_models, evaluate_regression
    from src.risk_classification import assign_risk_labels, train_risk_classifier, evaluate_classification
    from src.geospatial import load_geojson, csv_to_geodataframe, create_risk_map

    demo_csv = os.path.join(DATA_DIR, "demo_dataset.csv")
    demo_geojson = os.path.join(DATA_DIR, "demo_geospatial.geojson")

    # 1. Preprocessing
    df = load_data(demo_csv)
    df = clean_data(df)
    df = engineer_features(df)
    total_records = len(df)
    stations_count = df["station_id"].nunique()

    # 2. LSTM Validation
    stn_df = df[df["station_name"] == df["station_name"].iloc[0]].sort_values("date").reset_index(drop=True)
    avail_feat = [f for f in FEATURES if f in stn_df.columns]
    tr_df, ts_df = chronological_split(stn_df, TEST_SPLIT_RATIO)
    tr_norm, scaler = normalize_features(tr_df, avail_feat)
    ts_norm = ts_df.copy()
    ts_norm[avail_feat] = scaler.transform(ts_df[avail_feat].values)
    target_idx = avail_feat.index(TARGET_REGRESSION)

    X_tr, y_tr = create_sequences(tr_norm[avail_feat].values, tr_norm[avail_feat].values[:, target_idx], 5)
    X_ts, y_ts = create_sequences(ts_norm[avail_feat].values, ts_norm[avail_feat].values[:, target_idx], 5)

    lstm_model = build_lstm_model((5, len(avail_feat)), units=32)
    history = train_lstm_model(lstm_model, X_tr, y_tr, X_ts, y_ts, epochs=3, batch_size=32)
    lstm_preds = predict_lstm(lstm_model, X_ts)
    lstm_mae = float(np.mean(np.abs(y_ts - lstm_preds)))
    lstm_rmse = float(np.sqrt(np.mean((y_ts - lstm_preds) ** 2)))

    # 3. Benchmarking
    clean_stn = stn_df.dropna(subset=avail_feat + [TARGET_REGRESSION])
    b_tr, b_ts = chronological_split(clean_stn, TEST_SPLIT_RATIO)
    bench_df, _ = compare_models(
        b_tr[avail_feat].values, b_tr[TARGET_REGRESSION].values,
        b_ts[avail_feat].values, b_ts[TARGET_REGRESSION].values,
    )
    bench_results = bench_df.to_dict(orient="records")

    # 4. Classification
    clf_df = assign_risk_labels(df)
    c_tr, c_ts = chronological_split(clf_df, TEST_SPLIT_RATIO)
    clf = train_risk_classifier(c_tr[avail_feat].values, c_tr["flood_risk"].values, n_estimators=50)
    c_preds = clf.predict(c_ts[avail_feat].values)
    c_probs = clf.predict_proba(c_ts[avail_feat].values)
    clf_eval = evaluate_classification(c_ts["flood_risk"].values, c_preds, c_probs, labels=RISK_CATEGORIES)

    # 5. Geospatial
    gdf = load_geojson(demo_geojson)
    fmap = create_risk_map(gdf, risk_col="risk_zone" if "risk_zone" in gdf.columns else "flood_risk")
    has_map = fmap is not None

    duration = time.time() - start

    metrics = {
        "status": "PASSED",
        "duration_seconds": round(duration, 3),
        "data_records": total_records,
        "stations_count": stations_count,
        "neural_backend": "TensorFlow_Keras" if TF_AVAILABLE else "Temporal_Autoregression_Dual_Tier_Fallback",
        "lstm_forecast": {
            "test_mae": round(lstm_mae, 4),
            "test_rmse": round(lstm_rmse, 4),
            "epochs_evaluated": len(history.history.get("loss", [])),
        },
        "benchmarking_models": bench_results,
        "risk_classification": {
            "macro_precision": clf_eval["precision"],
            "macro_recall": clf_eval["recall"],
            "macro_f1": clf_eval["f1"],
            "macro_pr_auc": clf_eval["pr_auc"].get("macro_avg", 0.0),
        },
        "geospatial": {
            "features_loaded": len(gdf),
            "map_object_generated": has_map,
        },
    }

    print(f"ML Pipeline validation completed in {duration:.2f}s with status: PASSED")
    print(f"LSTM Test MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}")
    print(f"Classification Macro F1: {clf_eval['f1']:.4f}, Macro PR-AUC: {clf_eval['pr_auc'].get('macro_avg', 0.0):.4f}")
    return metrics


def main():
    start_total = time.time()
    timestamp = datetime.utcnow().isoformat() + "Z"

    print("==================================================================")
    print(f"FLOODGUARD AI TEST HARNESS // {timestamp}")
    print("==================================================================")
    print(f"Python: {platform.python_version()} on {platform.system()} ({platform.machine()})")
    print(f"Executable: {sys.executable}")
    print(f"Root: {PROJECT_ROOT}\n")

    syntax_res = run_syntax_checks()
    pytest_res = run_pytest_suite()
    ml_res = run_ml_pipeline_validation()

    duration_total = time.time() - start_total

    summary = {
        "project": "FloodGuard AI",
        "timestamp_utc": timestamp,
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "duration_seconds": round(duration_total, 3),
        "all_passed": (
            syntax_res["status"] == "PASSED"
            and pytest_res["status"] == "PASSED"
            and ml_res["status"] == "PASSED"
        ),
        "steps": {
            "syntax_audit": syntax_res,
            "pytest_suite": pytest_res,
            "ml_geospatial_pipeline": ml_res,
        },
    }

    # Save summary JSON
    summary_json_path = os.path.join(REPORTS_DIR, "test_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[REPORT] Saved JSON test summary to {summary_json_path}")

    # Save summary Markdown
    summary_md_path = os.path.join(REPORTS_DIR, "test_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# FloodGuard AI — Automated Test Execution Report\n\n")
        f.write(f"- **Execution Timestamp (UTC)**: `{timestamp}`\n")
        f.write(f"- **Operating System**: `{platform.platform()}`\n")
        f.write(f"- **Python Version**: `{platform.python_version()}`\n")
        f.write(f"- **Total Duration**: `{duration_total:.2f} seconds`\n")
        f.write(f"- **Overall Status**: **{'PASSED' if summary['all_passed'] else 'FAILED'}**\n\n")

        f.write(f"## 1. Syntax Audit\n")
        f.write(f"- Status: `{syntax_res['status']}`\n")
        f.write(f"- Duration: `{syntax_res['duration_seconds']}s`\n\n")

        f.write(f"## 2. Unit & Integration Tests (Pytest)\n")
        f.write(f"- Total Passed: `{pytest_res['passed_count']}`\n")
        f.write(f"- Total Failed: `{pytest_res['failed_count']}`\n")
        f.write(f"- Total Skipped: `{pytest_res['skipped_count']}`\n")
        f.write(f"- Duration: `{pytest_res['duration_seconds']}s`\n\n")

        f.write(f"## 3. Machine Learning & Geospatial Pipeline\n")
        f.write(f"- Neural Backend: `{ml_res['neural_backend']}`\n")
        f.write(f"- LSTM Test MAE: `{ml_res['lstm_forecast']['test_mae']}`\n")
        f.write(f"- LSTM Test RMSE: `{ml_res['lstm_forecast']['test_rmse']}`\n")
        f.write(f"- Classification Macro F1: `{ml_res['risk_classification']['macro_f1']}`\n")
        f.write(f"- Classification Macro PR-AUC: `{ml_res['risk_classification']['macro_pr_auc']}`\n")
        f.write(f"- Geospatial Features Verified: `{ml_res['geospatial']['features_loaded']}`\n\n")

        f.write(f"### Benchmark Estimators\n")
        f.write(f"| Model | MAE | RMSE |\n|---|---|---|\n")
        for b in ml_res["benchmarking_models"]:
            f.write(f"| {b['Model']} | {b['MAE']:.4f} | {b['RMSE']:.4f} |\n")

    print(f"[REPORT] Saved Markdown test summary to {summary_md_path}")

    if not summary["all_passed"]:
        print("\nERROR: Critical test steps failed!")
        sys.exit(1)

    print("\nSUCCESS: All test stages passed completely!")
    sys.exit(0)


if __name__ == "__main__":
    main()
