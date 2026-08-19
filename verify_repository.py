from pathlib import Path
import re, sys
ROOT = Path(__file__).resolve().parent
errors=[]; warnings=[]
for n in range(1,11):
    p=ROOT/f"bab{n:02d}"
    if not p.is_dir(): errors.append(f"Folder wajib tidak ada: {p.name}")
# reject known legacy generated-output names
legacy=[
    "bab07/output/output_9_1_belum_dilatih.png",
    "bab07/output/output_9_5_setelah_finetuning.png",
    "bab08/output/output_10_1_metrik_validation.png",
    "bab08/output/output_10_2_prediction_mask.png",
    "bab10/output/output_13_2_tradeoff.png",
    "bab10/output/output_13_3_calibration.png",
]
for rel in legacy:
    if (ROOT/rel).exists(): errors.append(f"Artefak legacy masih ada: {rel}")
# active runners known in compact edition
for n in (3,4,6,7,8,9,10):
    p=ROOT/f"bab{n:02d}"/f"run_all_bab{n:02d}.py"
    if not p.exists(): errors.append(f"Runner tidak ada: {p.relative_to(ROOT)}")
# public-release gate
license_text=(ROOT/'LICENSE').read_text(encoding='utf-8', errors='ignore')
if 'TO BE CONFIRMED' in license_text:
    warnings.append('Lisensi kode belum diputuskan; jangan ubah repository menjadi Public sebelum keputusan lisensi.')
print("CVAI BOOK — REPOSITORY STRUCTURE VERIFICATION")
print("="*58)
for w in warnings: print("WARNING:", w)
for e in errors: print("ERROR  :", e)
if errors:
    print(f"\nFAILED: {len(errors)} error")
    sys.exit(1)
print("\nSTRUCTURE CHECK PASSED")
