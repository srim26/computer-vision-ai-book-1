"""Clean-run Bab 7 Compact Edition."""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
scripts = [
    ROOT / "code" / "generate_dataset.py",
    ROOT / "code" / "train_fasterrcnn.py",
    ROOT / "code" / "04_visualize_before_after.py",
]
for script in scripts:
    print("\n" + "="*72)
    print("RUN:", script.name)
    print("="*72)
    subprocess.run([sys.executable, str(script)], cwd=ROOT/"code", check=True)

required = [
    ROOT / "output" / "fasterrcnn_bab07.pt",
    ROOT / "output" / "metrics_bab07.json",
    ROOT / "output" / "output_07_01_sebelum_training.png",
    ROOT / "output" / "output_07_03_setelah_training.png",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Output wajib belum terbentuk:\n" + "\n".join(missing))
print("\nBAB 7 CLEAN-RUN PASSED")
