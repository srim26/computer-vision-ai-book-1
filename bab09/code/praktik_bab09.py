"""Bab 9 Compact — citra medis sintetis, CT windowing, dan contoh measurement.

Tidak menggunakan data pasien. Array CT dibuat secara sintetis dan hanya ditujukan
untuk demonstrasi pedagogis.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"output"
OUT.mkdir(parents=True,exist_ok=True)
SEED=42

def window(hu_array, level, width):
    lower=level-width/2
    upper=level+width/2
    clipped=np.clip(hu_array,lower,upper)
    return ((clipped-lower)/(upper-lower)*255.0).astype(np.uint8)

def make_synthetic_ct():
    rng=np.random.default_rng(SEED)
    y,x=np.mgrid[-1:1:256j,-1:1:256j]
    body=(x*x/0.78**2+y*y/0.92**2)<=1
    lung1=((x+0.27)**2/0.22**2+y*y/0.48**2)<=1
    lung2=((x-0.27)**2/0.22**2+y*y/0.48**2)<=1
    bone=(x*x+y*y)<=0.09**2
    hu=np.full((256,256),-1000.0,np.float32)
    hu[body]=40
    hu[lung1|lung2]=-750
    hu[bone]=700
    hu += rng.normal(0,25,hu.shape).astype(np.float32)
    return hu

def main():
    hu=make_synthetic_ct()
    soft=window(hu,level=40,width=400)
    lung=window(hu,level=-600,width=1500)

    print("Shape CT sintetis:",hu.shape)
    print(f"Rentang HU: {hu.min():.1f} .. {hu.max():.1f}")
    print("Soft tissue window: level=40, width=400")
    print("Lung window       : level=-600, width=1500")

    fig=plt.figure(figsize=(5,5))
    plt.imshow(hu,cmap="gray"); plt.axis("off"); plt.tight_layout()
    p1=OUT/"output_09_01_ct_sintetis.png"
    plt.savefig(p1,dpi=180,bbox_inches="tight"); plt.close(fig)

    fig=plt.figure(figsize=(8,4))
    ax1=fig.add_axes([0.03,0.08,0.45,0.84])
    ax2=fig.add_axes([0.52,0.08,0.45,0.84])
    ax1.imshow(soft,cmap="gray",vmin=0,vmax=255); ax1.set_title("Soft tissue"); ax1.axis("off")
    ax2.imshow(lung,cmap="gray",vmin=0,vmax=255); ax2.set_title("Lung"); ax2.axis("off")
    p2=OUT/"output_09_02_windowing_ct.png"
    plt.savefig(p2,dpi=180,bbox_inches="tight"); plt.close(fig)

    # Contoh numerik measurement; bukan hasil segmentasi aktual pada run ini.
    marker_real_area_cm2=4.0
    marker_pixels=6400
    wound_pixels=7068
    area_scale=marker_real_area_cm2/marker_pixels
    estimated_area_cm2=wound_pixels*area_scale
    print(f"Skala contoh: {area_scale:.6f} cm²/pixel")
    print(f"Estimasi luas contoh: {estimated_area_cm2:.4f} cm²")
    print("Saved:",p1)
    print("Saved:",p2)

if __name__=="__main__":
    main()
