"""Clean-run Bab 6 Compact Edition."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
for name in ["00_prepare_oxford_pet.py","01_transfer_learning_mobilenetv2.py","02_plot_learning_curve.py"]:
    script=ROOT/"code"/name
    print("\n"+"="*72); print("RUN:",name); print("="*72)
    subprocess.run([sys.executable,str(script)],cwd=ROOT,check=True)
required=[ROOT/"output"/"metrics_bab06.json",ROOT/"output"/"mobilenetv2_pet.keras",
          ROOT/"output"/"output_06_01_learning_curve.png"]
missing=[str(p) for p in required if not p.exists()]
if missing: raise FileNotFoundError("\n".join(missing))
print("\nBAB 6 CLEAN-RUN PASSED")
