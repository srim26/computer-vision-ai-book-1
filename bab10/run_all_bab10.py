"""Clean-run Bab 10 Compact Edition."""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
script = ROOT / "code" / "01_evaluate_synthetic_medical_ai.py"

print("=" * 72)
print("RUN:", script.name)
print("=" * 72)
subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

required = [
    ROOT / "output" / "metrics_bab10.json",
    ROOT / "output" / "output_10_01_threshold_tradeoff.png",
    ROOT / "output" / "output_10_02_calibration_curve.png",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Output wajib belum terbentuk:\n" + "\n".join(missing))

print("\nBAB 10 CLEAN-RUN PASSED")
