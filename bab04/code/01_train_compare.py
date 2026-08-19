"""
Bab 4 - Machine Learning menuju Deep Learning
Pipeline lengkap: ekstraksi fitur (histogram warna 3D) -> split -> SVM -> Random Forest.
Sesuai kode yang ditunjukkan di naskah, dijalankan end-to-end untuk clean-run nyata.
"""
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import json


def ekstrak_fitur(image):
    image = cv2.resize(image, (128, 128))
    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


def main():
    X, y = [], []
    ROOT = Path(__file__).resolve().parent.parent
    dataset_path = ROOT / "dataset"
    for label in ["kelas_A", "kelas_B"]:
        folder = dataset_path / label
        for file in sorted(folder.glob("*")):
            image = cv2.imread(str(file))
            if image is None:
                continue
            X.append(ekstrak_fitur(image))
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print("Bentuk X:", X.shape)
    print("Bentuk y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])

    # === SVM ===
    svm_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", random_state=42)
    )
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(y_test, y_pred_svm, average="weighted")
    print("\n=== SVM ===")
    print("Accuracy:", acc_svm)
    print(classification_report(y_test, y_pred_svm))

    # === Random Forest ===
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average="weighted")
    print("\n=== Random Forest ===")
    print("Accuracy:", acc_rf)
    print(classification_report(y_test, y_pred_rf))

    results = {
        "seed": 42,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "svm": {"accuracy": acc_svm, "precision_weighted": prec_svm, "recall_weighted": rec_svm, "f1_weighted": f1_svm},
        "random_forest": {"accuracy": acc_rf, "precision_weighted": prec_rf, "recall_weighted": rec_rf, "f1_weighted": f1_rf},
    }
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "output_bab04_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nHasil disimpan ke output/output_bab04_result.json")

    # === Output 4.1: Confusion Matrix SVM ===
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_svm,
        display_labels=["kelas_A", "kelas_B"],
        values_format="d"
    )
    plt.title("Confusion Matrix — SVM")
    plt.tight_layout()
    cm_path = output_dir / "output_04_01_confusion_matrix_svm.png"
    plt.savefig(cm_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix SVM disimpan ke {cm_path}")


if __name__ == "__main__":
    main()
