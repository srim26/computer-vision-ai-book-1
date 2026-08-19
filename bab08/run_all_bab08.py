"""Clean-run Bab 8 Compact Edition."""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent
for name in ["01_generate_synthetic_dataset.py","02_train_unet.py","03_evaluate_and_generate_outputs.py"]:
    print("\n"+"="*72); print("RUN:",name); print("="*72)
    subprocess.run([sys.executable,str(ROOT/"code"/name)],cwd=ROOT,check=True)
required=[
 ROOT/"output"/"output_08_01_metrik_validation.png",
 ROOT/"output"/"output_08_02_prediction_mask.png",
]
missing=[str(p) for p in required if not p.exists()]
if missing: raise FileNotFoundError("Output wajib belum terbentuk:\n"+"\n".join(missing))
print("\nBAB 8 CLEAN-RUN PASSED")
