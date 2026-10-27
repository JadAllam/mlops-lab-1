"""Check a processed Food-11 folder against the data contract."""

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from food11.data import CLASSES, IMAGE_SIZE, SPLITS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    return parser.parse_args()


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def check_classes(root):
    ok = True
    for split in SPLITS:
        found = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
        if found != sorted(CLASSES):
            missing = sorted(set(CLASSES) - set(found))
            extra = sorted(set(found) - set(CLASSES))
            ok = report(f"classes in {split}", False, f"missing={missing} extra={extra}")
        else:
            ok = report(f"classes in {split}", True) and ok
    return ok


def check_images(root):
    bad = []
    for path in root.rglob("*.jpg"):
        try:
            with Image.open(path) as img:
                if img.size != (IMAGE_SIZE, IMAGE_SIZE) or img.mode != "RGB":
                    bad.append(f"{path} {img.size} {img.mode}")
        except (OSError, ValueError) as exc:
            bad.append(f"{path} unreadable: {exc}")
    return report("images are RGB 128x128", not bad, f"{len(bad)} bad" if bad else "")


def check_min_per_class(root, minimum):
    short = []
    for split in SPLITS:
        for name in CLASSES:
            n = len(list((root / split / name).glob("*.jpg")))
            if n < minimum:
                short.append(f"{split}/{name}={n}")
    return report(f"at least {minimum} per class", not short, " ".join(short))


def check_no_duplicates_across_splits(root):
    seen = defaultdict(set)
    for split in SPLITS:
        for path in (root / split).rglob("*.jpg"):
            seen[hashlib.md5(path.read_bytes()).hexdigest()].add(split)
    dupes = {h: s for h, s in seen.items() if len(s) > 1}
    detail = ""
    if dupes:
        first = next(iter(dupes.items()))
        detail = f"{len(dupes)} shared, e.g. {first[0][:8]} in {sorted(first[1])}"
    return report("no image in two splits", not dupes, detail)


def check_balance(root, tau):
    shares = {}
    for split in ("training", "validation"):
        counts = Counter(
            {name: len(list((root / split / name).glob("*.jpg"))) for name in CLASSES}
        )
        total = sum(counts.values()) or 1
        shares[split] = {name: counts[name] / total for name in CLASSES}

    drifted = [
        f"{name}={abs(shares['training'][name] - shares['validation'][name]):.3f}"
        for name in CLASSES
        if abs(shares["training"][name] - shares["validation"][name]) > tau
    ]
    return report(f"train/val class share within {tau}", not drifted, " ".join(drifted))


def main():
    args = parse_args()
    if not args.root.is_dir():
        sys.exit(f"no such dataset: {args.root}")

    print(f"validating {args.root}")
    results = [
        check_classes(args.root),
        check_images(args.root),
        check_min_per_class(args.root, args.min_per_class),
        check_no_duplicates_across_splits(args.root),
        check_balance(args.root, args.tau),
    ]
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
