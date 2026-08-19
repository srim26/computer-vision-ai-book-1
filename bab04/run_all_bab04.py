"""Clean-run Bab 4 Compact Edition."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
scripts = [
    ROOT / "code" / "00_generate_dataset.py",
    ROOT / "code" / "01_train_compare.py",
]

for script in scripts:
    print("\n" + "=" * 72)
    print("RUN:", script.name)
    print("=" * 72)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

required = [
    ROOT / "output" / "output_bab04_result.json",
    ROOT / "output" / "output_04_01_confusion_matrix_svm.png",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Output wajib belum terbentuk:\n" + "\n".join(missing))

print("\nBAB 4 CLEAN-RUN PASSED")
