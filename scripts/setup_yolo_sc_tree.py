"""
===============================================================================
Build the directory tree that YOLO requires
===============================================================================
Ultralytics locates a label file by replacing the last occurrence of '/images/'
in the image path with '/labels/'. The single-class labels produced by
generate_scenarios_single_class.py live in a separate directory, so a parallel
tree is built in which the two sit side by side:

  <root>/yolo_sc/images -> dataset/images
  <root>/yolo_sc/labels -> labels_single_class

The file lists under scenarios_single_class are then rewritten to point at this
new root (the originals are kept as .txt.orig) and the 'path' field of the
.yaml files is updated.

Usage:
    python setup_yolo_sc_tree.py          # all scenarios
    python setup_yolo_sc_tree.py S1 S2    # selected scenarios
===============================================================================
"""
import sys, shutil, yaml
from pathlib import Path

# Set this to the local installation.
ROOT = Path("/path/to/dolphin_study")

DATASET = ROOT / "dataset"
TREE = ROOT / "yolo_sc"
SCEN = ROOT / "scenarios_single_class"

OLD_PREFIX = str(DATASET / "images")
NEW_PREFIX = str(TREE / "images")


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)
    print(f"  {dst} -> {src}")


def main():
    only = sys.argv[1:] or None

    if not (ROOT / "labels_single_class").is_dir():
        sys.exit("ERROR: labels_single_class not found. Run "
                 "generate_scenarios_single_class.py first.")

    TREE.mkdir(parents=True, exist_ok=True)
    print("symbolic links:")
    link(DATASET / "images", TREE / "images")
    link(ROOT / "labels_single_class", TREE / "labels")

    print("\nfile lists:")
    for sd in sorted(SCEN.iterdir()):
        if not sd.is_dir() or (only and sd.name not in only):
            continue
        sc = sd.name
        for txt in sorted(sd.glob(f"{sc}_*.txt")):
            if txt.name.endswith(".orig"):
                continue
            orig = txt.with_suffix(".txt.orig")
            if not orig.exists():
                shutil.copy2(txt, orig)
            lines = orig.read_text().splitlines()
            new = [l.replace(OLD_PREFIX, NEW_PREFIX) for l in lines if l.strip()]
            txt.write_text("\n".join(new) + ("\n" if new else ""))
            bad = sum(1 for l in new if not l.startswith(NEW_PREFIX))
            print(f"  {sc}/{txt.name:22s} {len(new):6d} lines"
                  + (f"   !! {bad} lines not rewritten" if bad else ""))

        for y in sorted(sd.glob(f"{sc}_*.yaml")):
            d = yaml.safe_load(open(y))
            d["path"] = str(TREE)
            d["names"] = {0: "dolphin"}
            yaml.dump(d, open(y, "w"), default_flow_style=False)
        print(f"  {sc}: yaml path -> {TREE}")

    # check that the label files are actually found for a sample of images
    print("\nverification:")
    for sd in sorted(SCEN.iterdir()):
        if not sd.is_dir() or (only and sd.name not in only):
            continue
        sc = sd.name
        for split in ["train", "test_local", "test_ext"]:
            f = sd / f"{sc}_{split}.txt"
            if not f.exists():
                continue
            lines = [l for l in f.read_text().splitlines() if l.strip()]
            if not lines:
                print(f"  {sc}/{split}: empty")
                continue
            ok = 0
            for l in lines[:200]:
                lab = Path(l.replace("/images/", "/labels/")).with_suffix(".txt")
                if lab.is_file():
                    ok += 1
            print(f"  {sc}/{split:11s} label found for {ok} "
                  f"of the first 200 images")


main()
