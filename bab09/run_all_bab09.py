"""Clean-run Bab 9 Compact Edition."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
script=ROOT/"code"/"praktik_bab09.py"
print("="*72); print("RUN:",script.name); print("="*72)
subprocess.run([sys.executable,str(script)],cwd=ROOT,check=True)
required=[ROOT/"output"/"output_09_01_ct_sintetis.png",
          ROOT/"output"/"output_09_02_windowing_ct.png"]
missing=[str(p) for p in required if not p.exists()]
if missing: raise FileNotFoundError("\n".join(missing))
print("\nBAB 9 CLEAN-RUN PASSED")
