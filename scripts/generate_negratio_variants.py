"""
===============================================================================
Variants with different proportions of negative images
===============================================================================
Purpose: produce several versions of one scenario that differ only in how many
         of the training images contain no dolphins. The positive frames are
         held fixed and the negatives are subsampled.

Question: not "which kind of data is worth adding", which the scenario design
          addresses, but "how much negative material is needed".

Design decisions:
  - positive images (those carrying dolphin annotations) are never touched
  - negatives are subsampled in a stratified manner, so that each source
    subset keeps its share and only the quantity changes
  - the validation split is taken from each variant's own training set and
    therefore carries the same proportion
  - the test sets are left unchanged
  - the random seed is fixed, so the subsampling is reproducible

Output: <root>/scenarios_negratio/<S>_neg<NN>/
        The existing scenarios_single_class/ directory is left untouched.
===============================================================================
"""
import json, random, shutil, sys, argparse, collections
from pathlib import Path
import yaml

SEED = 42

# Set these to the local installation.
ROOT = Path("/path/to/dolphin_study")
SRC = ROOT / "scenarios_single_class"
OUT = ROOT / "scenarios_negratio"
TREE = ROOT / "yolo_sc"          # symlink tree used by YOLO, built beforehand


def split_pos_neg(coco):
    """Separate images into positive (annotated) and negative."""
    with_ann = {a["image_id"] for a in coco["annotations"]}
    pos = [i for i in coco["images"] if i["id"] in with_ann]
    neg = [i for i in coco["images"] if i["id"] not in with_ann]
    return pos, neg


def subsample_negatives(neg, target_n, rng):
    """Pick target_n negatives while preserving the share of each subset."""
    by_folder = collections.defaultdict(list)
    for im in neg:
        by_folder[im["file_name"].split("/")[0]].append(im)

    total = len(neg)
    picked, quotas = [], {}
    for f, items in sorted(by_folder.items()):
        quotas[f] = round(target_n * len(items) / total)

    # absorb the rounding difference in the largest subset
    diff = target_n - sum(quotas.values())
    if diff != 0 and quotas:
        biggest = max(quotas, key=lambda f: len(by_folder[f]))
        quotas[biggest] = max(0, quotas[biggest] + diff)

    for f, items in sorted(by_folder.items()):
        k = min(quotas[f], len(items))
        picked += rng.sample(sorted(items, key=lambda x: x["id"]), k)

    return picked, {f: min(quotas[f], len(by_folder[f])) for f in quotas}


def build_variant(coco, ratio, rng, label):
    """Build a new COCO file with the target proportion of negatives."""
    pos, neg = split_pos_neg(coco)
    n_pos = len(pos)
    # ratio = neg / (pos + neg)  ->  neg = pos * ratio / (1 - ratio)
    target_neg = round(n_pos * ratio / (1 - ratio))

    if target_neg > len(neg):
        sys.exit(f"ERROR [{label}]: {ratio:.0%} needs {target_neg} negatives, "
                 f"but only {len(neg)} are available. This scenario cannot "
                 f"support that proportion.")

    picked, quotas = subsample_negatives(neg, target_neg, rng)
    keep = pos + picked
    keep_ids = {i["id"] for i in keep}
    keep.sort(key=lambda x: x["id"])

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": keep,
        "annotations": [a for a in coco["annotations"]
                        if a["image_id"] in keep_ids],
    }
    actual = len(picked) / len(keep)
    return out, dict(pos=n_pos, neg=len(picked), total=len(keep),
                     actual_ratio=actual, quotas=quotas, available_neg=len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.05, 0.10, 0.30, 0.50])
    a = ap.parse_args()

    sd = SRC / a.scenario
    if not sd.is_dir():
        sys.exit(f"ERROR: {sd} not found")

    OUT.mkdir(parents=True, exist_ok=True)
    names = {0: "dolphin"}

    for ratio in a.ratios:
        tag = f"{a.scenario}_neg{int(round(ratio*100)):02d}"
        vd = OUT / tag
        vd.mkdir(parents=True, exist_ok=True)
        rng = random.Random(SEED)

        print(f"\n{'='*64}\n### {tag}   (target proportion {ratio:.0%})\n{'='*64}")

        for split in ["train", "val"]:
            src = sd / f"{a.scenario}_{split}.json"
            coco = json.load(open(src))
            new, st = build_variant(coco, ratio, rng, f"{tag}/{split}")

            json.dump(new, open(vd / f"{tag}_{split}.json", "w"), indent=4)

            paths = [str(ROOT / "dataset/images" / i["file_name"])
                     for i in new["images"]]
            # the file lists used by YOLO must point at the symlink tree
            paths = [p.replace(str(ROOT / "dataset/images"),
                               str(TREE / "images")) for p in paths]
            (vd / f"{tag}_{split}.txt").write_text("\n".join(sorted(paths)) + "\n")

            print(f"  {split:6s} positive={st['pos']:5d}  "
                  f"negative={st['neg']:5d}  total={st['total']:5d}  "
                  f"actual={st['actual_ratio']:.1%}  "
                  f"(negatives available: {st['available_neg']})")
            for f, q in sorted(st["quotas"].items()):
                print(f"           {f:22s} {q:5d}")

        # the test sets are copied unchanged
        for split in ["test_local", "test_ext"]:
            for ext in ["json", "txt"]:
                s = sd / f"{a.scenario}_{split}.{ext}"
                if s.is_file():
                    shutil.copy2(s, vd / f"{tag}_{split}.{ext}")
        print("  test_local / test_ext: copied unchanged")

        # YAML files for YOLO
        for lbl, key in [("local", "test_local"), ("ext", "test_ext")]:
            content = {
                "path": str(TREE),
                "train": str((vd / f"{tag}_train.txt").resolve()),
                "val": str((vd / f"{tag}_val.txt").resolve()),
                "test": str((vd / f"{tag}_{key}.txt").resolve()),
                "names": names,
            }
            yaml.dump(content, open(vd / f"{tag}_{lbl}.yaml", "w"),
                      default_flow_style=False)

    print(f"\n{'='*64}")
    print(f"output: {OUT}")
    print("\nnext step, build a COCO view for each variant:")
    for ratio in a.ratios:
        tag = f"{a.scenario}_neg{int(round(ratio*100)):02d}"
        print(f"  python make_coco_view.py --scenario {tag} \\")
        print(f"    --scenarios-root {OUT} \\")
        print(f"    --out-root {ROOT}/coco_views_negratio")


main()
