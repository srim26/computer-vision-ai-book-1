"""Clean-run Bab 3 Compact Edition."""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
scripts = [
    ROOT / "code" / "00_generate_synthetic_scene.py",
    ROOT / "code" / "01_image_processing.py",
    ROOT / "code" / "02_extract_features.py",
    ROOT / "code" / "03_generate_feature_output.py",
]
for script in scripts:
    print("\n" + "="*72)
    print("RUN:", script.name)
    print("="*72)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
print("\nBAB 3 CLEAN-RUN PASSED")
