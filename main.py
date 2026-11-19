"""Run the pipeline: EDA -> backtest -> backtest chart -> diagrams.  Usage: python main.py [--skip-backtest]"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
steps = ["eda.py", "backtest.py", "plot_backtest.py", "diagrams.py"]
if "--skip-backtest" in sys.argv and (SRC.parent / "outputs/backtest/scores_per_series.csv").exists():
    steps.remove("backtest.py")
for step in steps:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
