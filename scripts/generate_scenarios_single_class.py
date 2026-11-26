"""
===============================================================================
Scenario splitter, single-class (dolphin-only) setup
===============================================================================
Purpose: turn the "other" category into negative examples rather than a second
         detection class. Frames showing other species remain in the training
         and test sets, but their annotations are dropped, so the model learns
         to treat them as background.

Reason : because of how the data had to be collected, no frame contains both
         categories and the "other" class is never evaluated in-domain in any
         scenario. The question asked here is therefore not "is this fish a
         dolphin or something else" but "is a dolphin present" together with
         "does the detector raise false alarms on species it has not seen".

Output : <dataset_root>/../scenarios_single_class/<S>/
         containing <S>_train/val/test_local/test_ext as .txt and .json.
         An existing scenarios/ directory is left untouched.

Note   : YOLO requires a .txt label file for every image. Frames that contain
         no dolphin get an empty file, which Ultralytics treats as background.
         A separate label directory is written for this.

Usage:
    python generate_scenarios_single_class.py          # all scenarios
    python generate_scenarios_single_class.py S1 S2    # selected scenarios
===============================================================================
"""
import json, random, yaml, sys
from pathlib import Path

RANDOM_SEED = 42
VAL_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DOLPHIN_ID = 0          # dolphin category_id in the master annotation file

# Set this to the scenario definition file of the local installation.
CONFIG = Path("/path/to/Dolphin-Detector/configs/scenarios.yaml")


def collect(dataset_root, folders):
    out, missing = [], []
    for name in folders:
        p = dataset_root / "images" / name
        if not p.exists():
            missing.append(name); continue
        out += [str(f.resolve()) for f in p.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(out), missing


def filter_coco_single(master, image_paths):
    """Keep the images, but only the dolphin annotations."""
    targets = {f"{Path(p).parent.name}/{Path(p).name}" for p in image_paths}
    imgs = [i for i in master["images"] if i["file_name"] in targets]
    ids = {i["id"] for i in imgs}
    anns = [a for a in master["annotations"]
            if a["image_id"] in ids and a["category_id"] == DOLPHIN_ID]
    dropped = sum(1 for a in master["annotations"]
                  if a["image_id"] in ids and a["category_id"] != DOLPHIN_ID)
    coco = {
        "info": master.get("info", {}),
        "licenses": master.get("licenses", []),
        "categories": [{"id": DOLPHIN_ID, "name": "dolphin",
                        "supercategory": "marine_object"}],
        "images": imgs,
        "annotations": anns,
    }
    return coco, dropped


def labels_from_annotations(dataset_root, out_labels, master):
    """Write YOLO label files directly from the master annotation file.

    Used when no pre-existing label directory is available. Boxes are converted
    from COCO (x, y, w, h in pixels) to YOLO (normalised centre and size).
    """
    size = {i["id"]: (i["width"], i["height"]) for i in master["images"]}
    name = {i["id"]: i["file_name"] for i in master["images"]}
    boxes = {}
    for a in master["annotations"]:
        if a["category_id"] != DOLPHIN_ID:
            continue
        boxes.setdefault(a["image_id"], []).append(a["bbox"])

    n_box = n_blank = 0
    for d in sorted((dataset_root / "images").iterdir()):
        if not d.is_dir():
            continue
        (out_labels / d.name).mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dst = out_labels / d.name / f"{f.stem}.txt"
            rel = f"{d.name}/{f.name}"
            iid = next((i for i, n in name.items() if n == rel), None)
            lines = []
            if iid is not None and iid in boxes:
                w_img, h_img = size[iid]
                for x, y, w, h in boxes[iid]:
                    lines.append(
                        f"{DOLPHIN_ID} {(x + w/2)/w_img:.6f} "
                        f"{(y + h/2)/h_img:.6f} {w/w_img:.6f} {h/h_img:.6f}")
            if lines:
                dst.write_text("\n".join(lines) + "\n"); n_box += 1
            else:
                dst.write_text(""); n_blank += 1
    print(f"\nYOLO label directory: {out_labels}  (built from annotations)")
    print(f"  with dolphin boxes: {n_box}   empty (negative): {n_blank}")


def write_labels(dataset_root, out_labels, master):
    """Copy label files that contain dolphin boxes, write empty ones for the
    rest. Falls back to building the labels from the annotation file if no
    label directory is present."""
    src_labels = dataset_root / "labels"
    if not src_labels.exists():
        print("[INFO] no labels/ directory found, "
              "deriving YOLO labels from the annotation file")
        labels_from_annotations(dataset_root, out_labels, master)
        return

    by_name = {i["file_name"]: i["id"] for i in master["images"]}
    has_dolphin = {a["image_id"] for a in master["annotations"]
                   if a["category_id"] == DOLPHIN_ID}

    n_copy = n_blank = 0
    for d in sorted((dataset_root / "images").iterdir()):
        if not d.is_dir():
            continue
        (out_labels / d.name).mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = f"{d.name}/{f.name}"
            dst = out_labels / d.name / f"{f.stem}.txt"
            src = src_labels / d.name / f"{f.stem}.txt"
            if rel in by_name and by_name[rel] in has_dolphin and src.exists():
                # keep only the dolphin rows (class 0)
                lines = [l for l in src.read_text().splitlines()
                         if l.strip() and l.split()[0] == str(DOLPHIN_ID)]
                dst.write_text("\n".join(lines) + ("\n" if lines else ""))
                n_copy += 1
            else:
                dst.write_text("")
                n_blank += 1
    print(f"\nYOLO label directory: {out_labels}")
    print(f"  with dolphin boxes: {n_copy}   empty (negative): {n_blank}")


def main():
    random.seed(RANDOM_SEED)
    cfg = yaml.safe_load(open(CONFIG))
    dataset_root = Path(cfg["dataset_root"])
    out_root = dataset_root.parent / "scenarios_single_class"
    out_root.mkdir(parents=True, exist_ok=True)
    out_labels = dataset_root.parent / "labels_single_class"

    master = json.load(open(dataset_root / "annotations.json"))
    names = {0: "dolphin"}

    only = sys.argv[1:] or None
    problems = []

    for sc, s in cfg["scenarios"].items():
        if only and sc not in only:
            continue
        print(f"\n--- {sc} ---")
        folder = out_root / sc
        folder.mkdir(parents=True, exist_ok=True)

        train_raw, miss = collect(dataset_root, s["train_folders"])
        if miss:
            problems.append((sc, "train", miss))
        random.shuffle(train_raw)
        n_val = int(len(train_raw) * VAL_RATIO)
        splits = {"val": train_raw[:n_val], "train": train_raw[n_val:]}
        for key, yk in [("test_local", "test_folders_local"),
                        ("test_ext", "test_folders_ext")]:
            imgs, miss = collect(dataset_root, s.get(yk, []))
            if miss:
                problems.append((sc, key, miss))
            splits[key] = imgs

        for key in ["train", "val", "test_local", "test_ext"]:
            imgs = splits[key]
            (folder / f"{sc}_{key}.txt").write_text(
                "\n".join(imgs) + "\n" if imgs else "")
            coco, dropped = filter_coco_single(master, imgs)
            json.dump(coco, open(folder / f"{sc}_{key}.json", "w"), indent=4)
            n_pos = len({a["image_id"] for a in coco["annotations"]})
            n_neg = len(coco["images"]) - n_pos
            print(f"  {key:11s} img={len(imgs):6d}  "
                  f"dolphin_ann={len(coco['annotations']):6d}  "
                  f"positive_img={n_pos:5d}  negative_img={n_neg:5d}  "
                  f"(dropped other_ann={dropped})")
            if imgs and len(imgs) != len(coco["images"]):
                problems.append((sc, key,
                                 f"{len(imgs)-len(coco['images'])} images "
                                 f"not found in the master annotation file"))

        for tag, key in [("local", "test_local"), ("ext", "test_ext")]:
            content = {
                "path": str(dataset_root.resolve()),
                "train": str((folder / f"{sc}_train.txt").resolve()),
                "val": str((folder / f"{sc}_val.txt").resolve()),
                "test": str((folder / f"{sc}_{key}.txt").resolve()),
                "names": names,
            }
            with open(folder / f"{sc}_{tag}.yaml", "w") as f:
                yaml.dump(content, f, default_flow_style=False)

    write_labels(dataset_root, out_labels, master)

    print("\n" + "=" * 60)
    if problems:
        print("!!! PROBLEMS:")
        for r in problems:
            print("   ", r)
    else:
        print("no problems found")
    print(f"\nJSON  -> {out_root}")
    print(f"LABEL -> {out_labels}")


main()
