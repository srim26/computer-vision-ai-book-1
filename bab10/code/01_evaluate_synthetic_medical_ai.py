"""Bab 10 Compact — evaluasi Medical AI dengan data sintetis transparan.

Tujuan:
1. Menunjukkan trade-off sensitivity/specificity pada beberapa threshold.
2. Menunjukkan calibration curve dari probabilitas prediksi sintetis.
3. Menyimpan metrik ke JSON.

CATATAN:
- Seluruh data dibuat secara sintetis dengan seed tetap.
- Ini bukan hasil model klinis dan bukan validasi medis.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

def make_synthetic_predictions(n=240):
    rng = np.random.default_rng(SEED)

    # 25% positif, 75% negatif untuk meniru situasi klasifikasi tidak seimbang.
    y_true = np.zeros(n, dtype=int)
    pos_idx = rng.choice(n, size=n // 4, replace=False)
    y_true[pos_idx] = 1

    # Probabilitas sintetis: kelas positif cenderung lebih tinggi,
    # kelas negatif cenderung lebih rendah, tetapi masih overlap.
    probs = np.empty(n, dtype=float)
    probs[y_true == 1] = rng.beta(7, 3, size=(y_true == 1).sum())
    probs[y_true == 0] = rng.beta(2, 7, size=(y_true == 0).sum())

    # Sedikit "shrink" ke tengah untuk menghindari confidence ekstrem.
    probs = 0.05 + 0.90 * probs
    return y_true, probs

def safe_div(a, b):
    return float(a / b) if b else 0.0

def metrics_at_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    accuracy = safe_div(tp + tn, len(y_true))
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
    }

def generate_tradeoff(y_true, probs):
    thresholds = np.arange(0.10, 0.91, 0.05)
    rows = [metrics_at_threshold(y_true, probs, t) for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.plot(thresholds, [r["sensitivity"] for r in rows], marker="o", label="Sensitivity")
    ax.plot(thresholds, [r["specificity"] for r in rows], marker="s", label="Specificity")
    ax.axvline(0.50, linestyle="--", linewidth=1, label="Threshold = 0,50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.03)
    ax.set_title("Trade-off Sensitivity vs Specificity")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = OUT / "output_10_01_threshold_tradeoff.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return rows, path

def generate_calibration(y_true, probs):
    prob_true, prob_pred = calibration_curve(
        y_true, probs, n_bins=8, strategy="quantile"
    )
    brier = float(brier_score_loss(y_true, probs))

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=180)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Kalibrasi ideal")
    ax.plot(prob_pred, prob_true, marker="o", label=f"Data sintetis (Brier={brier:.3f})")
    ax.set_xlabel("Probabilitas prediksi rata-rata")
    ax.set_ylabel("Fraksi positif aktual")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    path = OUT / "output_10_02_calibration_curve.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return {
        "brier_score": brier,
        "mean_predicted_probability": [float(x) for x in prob_pred],
        "fraction_positive": [float(x) for x in prob_true],
    }, path

def main():
    y_true, probs = make_synthetic_predictions()

    tradeoff, p1 = generate_tradeoff(y_true, probs)
    calibration, p2 = generate_calibration(y_true, probs)
    t05 = metrics_at_threshold(y_true, probs, 0.50)

    result = {
        "seed": SEED,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "threshold_0_50": t05,
        "tradeoff": tradeoff,
        "calibration": calibration,
        "note": "Synthetic pedagogical data; not clinical validation."
    }

    metrics_path = OUT / "metrics_bab10.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Data sintetis:", len(y_true), "sampel")
    print("Positif:", int(y_true.sum()), "| Negatif:", int((1-y_true).sum()))
    print("Threshold 0.50:")
    print(json.dumps(t05, indent=2))
    print(f"Brier score: {calibration['brier_score']:.4f}")
    print("Saved:", p1)
    print("Saved:", p2)
    print("Saved:", metrics_path)

if __name__ == "__main__":
    main()
