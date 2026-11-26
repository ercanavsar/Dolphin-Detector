#!/usr/bin/env python3
"""
===============================================================================
Build the COCO directory layout expected by DINO and RT-DETRv2
===============================================================================
Converts the scenario JSON files into the directory structure the two
transformer-based frameworks expect. The original files are left untouched and
the images are not copied; a symbolic link is used instead.

Three views are produced per scenario:
  <out>/<SC>/             train2017=train    val2017=val         -> training
  <out>/<SC>_TEST_LOCAL/  train2017=train    val2017=test_local  -> local test
  <out>/<SC>_TEST_EXT/    train2017=train    val2017=test_ext    -> external test

Train and val are not merged; which image belongs to which split is determined
by the JSON files. Scenarios S9 to S16 have no local test set, so that view is
skipped for them.
===============================================================================
"""
import json, sys, argparse
from pathlib import Path

# Set these to the local installation.
SCENARIOS_ROOT = "/path/to/dolphin_study/scenarios_single_class"
IMAGES_ROOT = "/path/to/dolphin_study/dataset/images"
OUT_ROOT = "/path/to/dolphin_study/coco_views_single_class"

SHIFT = 1  # category_id 0 -> 1, leaving 0 for background


def convert(src_json, dst_json, images_root):
    d = json.load(open(src_json))
    for c in d["categories"]:
        c["id"] += SHIFT
    for a in d["annotations"]:
        a["category_id"] += SHIFT
        a.setdefault("iscrowd", 0)
        a.setdefault("segmentation", [])
        if "area" not in a:
            a["area"] = a["bbox"][2] * a["bbox"][3]
    missing = [im["file_name"] for im in d["images"]
               if not (images_root / im["file_name"]).is_file()]
    d["categories"].sort(key=lambda c: c["id"])
    json.dump(d, open(dst_json, "w"))
    present = sorted({a["category_id"] for a in d["annotations"]})
    return dict(images=len(d["images"]), anns=len(d["annotations"]),
                present=present, missing=missing,
                max_cat_id=max(c["id"] for c in d["categories"]))


def build_scenario(sc, scen_root, img_root, out_root):
    sd = scen_root / sc
    if not sd.is_dir():
        sys.exit(f"ERROR: scenario directory not found -> {sd}")

    views = {
        out_root / sc:                  [("train2017", "train"), ("val2017", "val")],
        out_root / f"{sc}_TEST_LOCAL":  [("train2017", "train"), ("val2017", "test_local")],
        out_root / f"{sc}_TEST_EXT":    [("train2017", "train"), ("val2017", "test_ext")],
    }

    max_cat, problems = 0, []
    for view_dir, entries in views.items():
        # skip this view if one of its splits is empty
        skip = False
        for _, split in entries:
            src = sd / f"{sc}_{split}.json"
            if not src.is_file():
                problems.append((sc, split, "file not found")); skip = True
            elif len(json.load(open(src))["images"]) == 0:
                print(f"\n### {view_dir.name}: {split} is empty, view skipped")
                skip = True
        if skip:
            continue

        (view_dir / "annotations").mkdir(parents=True, exist_ok=True)
        print(f"\n### {view_dir}")
        for imgdir, split in entries:
            src = sd / f"{sc}_{split}.json"
            link = view_dir / imgdir
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img_root)
            r = convert(src, view_dir / "annotations" / f"instances_{imgdir}.json",
                        img_root)
            max_cat = max(max_cat, r["max_cat_id"])
            print(f"  {imgdir:10s} <- {src.name}")
            print(f"      images={r['images']:6d}  anns={r['anns']:6d}  "
                  f"categories_present={r['present']}")
            if r["missing"]:
                problems.append((sc, split,
                                 f"{len(r['missing'])} files missing on disk"))
                print(f"      !!! {len(r['missing'])} FILES MISSING ON DISK")
    return max_cat, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scenarios-root", default=SCENARIOS_ROOT)
    ap.add_argument("--images-root", default=IMAGES_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    scen_root = Path(args.scenarios_root)
    img_root = Path(args.images_root).resolve()
    out_root = Path(args.out_root)
    if not img_root.is_dir():
        sys.exit(f"ERROR: image directory not found -> {img_root}")

    def key(p):
        s = p.name
        return (0, int(s[1:])) if s[1:].isdigit() else (1, s)

    if args.all:
        scenarios = [p.name for p in sorted(
            (x for x in scen_root.iterdir() if x.is_dir()), key=key)]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        avail = sorted(p.name for p in scen_root.iterdir() if p.is_dir())
        sys.exit(f"--scenario or --all is required. "
                 f"Available: {' '.join(avail)}")

    print(f"processing: {' '.join(scenarios)}")
    overall_max, all_problems = 0, []
    for sc in scenarios:
        m, p = build_scenario(sc, scen_root, img_root, out_root)
        overall_max = max(overall_max, m); all_problems += p

    print("\n" + "=" * 60)
    print(f"==> num_classes = {overall_max + 1}")
    if all_problems:
        print("!!! PROBLEMS:")
        for row in all_problems:
            print("   ", row)
    else:
        print("==> no problems found")


main()
