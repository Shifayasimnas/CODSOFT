from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eda import save_eda_plots


if __name__ == "__main__":
    summary = save_eda_plots()
    print("EDA completed.")
    print(summary)
