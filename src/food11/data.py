"""Build food11_processed and food11_processed_mini from food11_raw."""

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

SPLITS = ["training", "evaluation", "validation"]
IMAGE_SIZE = 128
MINI_PER_CLASS = 100

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "food11_raw"
PROCESSED_DIR = DATA_DIR / "food11_processed"
MINI_DIR = DATA_DIR / "food11_processed_mini"


def category_from_filename(filename):
    """'3_120.jpg' -> 'Egg'"""
    return CLASSES[int(Path(filename).stem.split("_")[0]) - 1]


def resize_image(img, size=IMAGE_SIZE):
    """Returns an RGB image of size x size."""
    return img.convert("RGB").resize((size, size), Image.BILINEAR)


def resize_one(job):
    src, dst = job
    with Image.open(src) as img:
        resize_image(img).save(dst, quality=95)


def build_split(split):
    src_split = RAW_DIR / split
    images = sorted(p for p in src_split.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    by_class = {name: [] for name in CLASSES}
    for path in images:
        by_class[category_from_filename(path.name)].append(path)

    jobs = []
    mini_pairs = []
    for name, paths in by_class.items():
        full_dir = PROCESSED_DIR / split / name
        mini_dir = MINI_DIR / split / name
        full_dir.mkdir(parents=True, exist_ok=True)
        mini_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(paths):
            dst = full_dir / src.name
            jobs.append((src, dst))
            if i < MINI_PER_CLASS:
                mini_pairs.append((dst, mini_dir / src.name))

    with ProcessPoolExecutor() as pool:
        list(pool.map(resize_one, jobs, chunksize=64))

    for processed, mini in mini_pairs:
        shutil.copyfile(processed, mini)

    print(f"{split}: {len(jobs)} processed, {len(mini_pairs)} mini")


def main():
    for split in SPLITS:
        build_split(split)


if __name__ == "__main__":
    main()
