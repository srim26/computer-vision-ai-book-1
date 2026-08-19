"""Plot learning curves dari history nyata Bab 6."""
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"output"
m=json.loads((OUT/"metrics_bab06.json").read_text(encoding="utf-8"))
h1=m["history_feature_extraction"]; h2=m["history_fine_tuning"]
acc=h1["accuracy"]+h2["accuracy"]; val=h1["val_accuracy"]+h2["val_accuracy"]
epochs=range(1,len(acc)+1)
plt.figure(figsize=(7,4.5))
plt.plot(epochs,acc,label="Training")
plt.plot(epochs,val,label="Validation")
plt.axvline(10.5,linestyle="--",linewidth=1)
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
p=OUT/"output_06_01_learning_curve.png"
plt.savefig(p,dpi=180,bbox_inches="tight"); plt.close()
print("Learning curve disimpan ke",p)
