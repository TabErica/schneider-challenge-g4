"""Run the pipeline: preprocess -> EDA -> risk EDA -> backtest -> sensitivity -> backtest chart -> diagrams.

Usage: python main.py [--origin YYYY-MM] [--skip-backtest] [--skip-sensitivity]
  --origin         forecast origin (default 2026-09, the data cutoff: the submission). An earlier origin (e.g. 2026-06)
                   is a held-back check: only preprocess + backtest run, the backtest sees data as of that origin and
                   writes outputs/backtest/origin_YYYY-MM/ (holdout_check.csv compares with the months after it).
  --skip-backtest     reuse existing backtest and sensitivity results (default origin only).
  --skip-sensitivity  skip the sensitivity step. Without it, sensitivity.py re-runs a variant only when its inputs
                      (panel, BOM workbook, backtest code, arguments) changed; otherwise it reuses the cached scores.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
CUTOFF = "2026-09"  # = preprocess.CUTOFF

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--origin", default=CUTOFF)
ap.add_argument("--skip-backtest", action="store_true")
ap.add_argument("--skip-sensitivity", action="store_true")
a = ap.parse_args()
if a.origin > CUTOFF:
    sys.exit(f"--origin {a.origin} is after the data cutoff {CUTOFF}")

if a.origin == CUTOFF:
    steps = [["preprocess.py"], ["eda.py"], ["eda_risk.py"], ["backtest.py"], ["sensitivity.py"],
             ["plot_backtest.py"], ["diagrams.py"]]
    if a.skip_backtest and (SRC.parent / "outputs/backtest/scores_per_series.csv").exists():
        steps = [s for s in steps if s[0] not in ("backtest.py", "sensitivity.py")]
    if a.skip_sensitivity:
        steps = [s for s in steps if s[0] != "sensitivity.py"]
else:  # held-back check: forecasting path only
    steps = [["preprocess.py"], ["backtest.py", "--origin", a.origin]]
for step in steps:
    print(f"\n=== {' '.join(step)} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / step[0]), *step[1:]], check=True, cwd=SRC)
