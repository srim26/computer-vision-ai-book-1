"""Bab 6 Compact — siapkan subset Oxford-IIIT Pet deterministik.

Target naskah:
- kelas: Abyssinian dan american_bulldog
- 200 citra per kelas
- split per kelas: 140 train, 30 validation, 30 test
"""
from pathlib import Path
import random, shutil, tarfile, urllib.request

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
SPLIT = DATA / "pet_subset"
URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ARCHIVE = DATA / "images.tar.gz"
CLASSES = ["Abyssinian", "american_bulldog"]
SEED = 42

def download():
    DATA.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        print("Archive ditemukan:", ARCHIVE)
        return
    print("Mengunduh Oxford-IIIT Pet images...")
    try:
        urllib.request.urlretrieve(URL, ARCHIVE)
    except Exception as e:
        raise RuntimeError(
            "Download otomatis gagal. Unduh manual images.tar.gz dari situs resmi "
            "Oxford-IIIT Pet dan simpan sebagai bab06/data/images.tar.gz."
        ) from e

def extract():
    images_dir = RAW / "images"
    if images_dir.exists():
        return images_dir
    RAW.mkdir(parents=True, exist_ok=True)
    print("Mengekstrak dataset...")
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(RAW)
    return images_dir

def select_files(images_dir, cls):
    files = sorted(images_dir.glob(f"{cls}_*.jpg"))
    if len(files) < 200:
        raise RuntimeError(f"{cls}: hanya {len(files)} citra; minimal 200 diperlukan.")
    rng = random.Random(SEED)
    rng.shuffle(files)
    return files[:200]

def main():
    download()
    images_dir = extract()
    if SPLIT.exists():
        shutil.rmtree(SPLIT)
    plan = [("train",140),("val",30),("test",30)]
    for cls in CLASSES:
        chosen=select_files(images_dir, cls)
        pos=0
        for split,n in plan:
            target=SPLIT/split/cls
            target.mkdir(parents=True, exist_ok=True)
            for f in chosen[pos:pos+n]:
                shutil.copy2(f,target/f.name)
            pos += n
    print("Subset siap:", SPLIT)
    for split,n in plan:
        print(f"{split:5s}: {n*2} citra ({n} per kelas)")

if __name__=="__main__":
    main()
