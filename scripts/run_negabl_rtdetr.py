#!/usr/bin/env python3
"""
===============================================================================
Proportion of negative images, RT-DETRv2, repeated over several seeds
===============================================================================
Trains and evaluates the four proportions (5, 10, 30 and 50 per cent) with ten
different random seeds. The repetition serves two purposes: to measure the
effect of the proportion itself, and to estimate how much two runs of the same
configuration differ, so that an observed difference can be judged against
that variation.

Isolation: this script never writes into the main results.
  configuration -> configs/rtdetrv2/negabl_<variant>_seed<N>.yml
  output        -> <EXP_ROOT>/<variant>_seed<N>/
The script stops if the output directory would overlap with the directory
holding the main experiments.

Usage:
  python run_negabl_rtdetr.py --dry-run           # show what would be run
  python run_negabl_rtdetr.py --variant neg05 --seed 1
  python run_negabl_rtdetr.py --all               # all forty runs (long)
  python run_negabl_rtdetr.py --all --eval-only   # skip the training
===============================================================================
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent

# Set these to the local installation.
RT_ROOT = Path("/path/to/RT-DETR/rtdetrv2_pytorch")
TRAIN_CFG = Path("/path/to/Dolphin-Detector/configs/train_config.yaml")
VIEWS = Path("/path/to/dolphin_study/coco_views_single_class")
EXP_ROOT = "/path/to/dolphin_study/experiments_negabl"

# The directory holding the main experiments, which must not be written to.
PROTECTED = Path("/path/to/dolphin_study/experiments").resolve()

RTCFG = RT_ROOT / "configs/rtdetrv2"
PRETRAIN = RT_ROOT / "weights/rtdetrv2_r50vd_6x_coco_ema.pth"

VARIANTS = ["neg05", "neg10", "neg30", "neg50"]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_CLASSES = 2


def sh(cmd, cwd=None, dry=False):
    line = " ".join(str(c) for c in cmd)
    print(f"\n$ {line}\n", flush=True)
    if dry:
        return
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        sys.exit(f"ERROR: command failed (exit {r.returncode})")


def check_isolation(exp_root):
    root = Path(exp_root).resolve()
    if root == PROTECTED or PROTECTED in root.parents or root in PROTECTED.parents:
        sys.exit(f"ERROR: the output directory overlaps with the main "
                 f"experiments.\n  requested : {root}\n"
                 f"  protected : {PROTECTED}")


def write_config(variant, seed, exp_root, epochs, batch_size):
    """Write a configuration file for this run, leaving the others untouched."""
    g = yaml.safe_load(open(TRAIN_CFG))
    gl, rt = g["global"], g["models"]["rtdetr"]
    view = VIEWS / f"S2_{variant}"
    if not view.is_dir():
        sys.exit(f"ERROR: view not found -> {view}")

    out_dir = Path(exp_root) / f"{variant}_seed{seed}"
    cfg = {
        "__include__": [
            "../dataset/coco_detection.yml",
            "../runtime.yml",
            "./include/dataloader.yml",
            "./include/optimizer.yml",
            "./include/rtdetrv2_r50vd.yml",
        ],
        "num_classes": NUM_CLASSES,
        "remap_mscoco_category": False,
        "RTDETRTransformerv2": {"num_classes": NUM_CLASSES},
        "Criterionv2": {"num_classes": NUM_CLASSES},
        "seed": seed,
        "epoches": epochs or gl["max_epochs"],
        "output_dir": str(out_dir),
        "use_ema": False,
        "use_amp": False,
        "checkpoint_freq": 10000,          # do not keep intermediate checkpoints
        "optimizer": {
            "type": rt["optimizer"],
            "lr": rt["lr0"],
            "betas": [0.9, 0.999],
            "weight_decay": rt["weight_decay"],
            "params": [
                {"params": "^(?=.*backbone)(?!.*norm).*$",
                 "lr": rt["lr_backbone"]},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$",
                 "weight_decay": 0.0},
            ],
        },
        "train_dataloader": {
            "dataset": {
                "img_folder": str(view / "train2017"),
                "ann_file": str(view / "annotations/instances_train2017.json"),
            },
            "total_batch_size": batch_size or gl["batch_size"],
            "num_workers": gl["workers"],
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(view / "val2017"),
                "ann_file": str(view / "annotations/instances_val2017.json"),
            },
            "total_batch_size": batch_size or gl["batch_size"],
            "num_workers": gl["workers"],
        },
    }

    path = RTCFG / f"negabl_{variant}_seed{seed}.yml"
    yaml.dump(cfg, open(path, "w"), default_flow_style=False, sort_keys=False)
    return path, out_dir


def run_one(variant, seed, a):
    cfg_path, exp = write_config(variant, seed, a.exp_root,
                                 a.epochs, a.batch_size)
    print(f"\n{'=' * 70}\n RT-DETRv2  {variant}  seed={seed}\n"
          f" config : {cfg_path}\n output : {exp}\n{'=' * 70}")

    if cfg_path.name != f"negabl_{variant}_seed{seed}.yml":
        sys.exit(f"ERROR: unexpected configuration name: {cfg_path.name}")
    if exp.name != f"{variant}_seed{seed}":
        sys.exit(f"ERROR: unexpected output directory: {exp}")

    if not a.eval_only:
        if not PRETRAIN.is_file():
            sys.exit(f"ERROR: COCO checkpoint not found -> {PRETRAIN}")
        sh([sys.executable, "tools/train.py",
            "-c", cfg_path.relative_to(RT_ROOT),
            "-t", PRETRAIN.relative_to(RT_ROOT),
            "--seed", seed], cwd=RT_ROOT, dry=a.dry_run)

    for tag in ["LOCAL", "EXT"]:
        view = VIEWS / f"S2_{variant}_TEST_{tag}"
        if not view.is_dir():
            print(f"\n[SKIP] {tag}: view not found ({view.name})")
            continue
        for wtag, wname in [("best", "best.pth"), ("last", "last.pth")]:
            ckpt = exp / wname
            if not a.dry_run and not ckpt.is_file():
                print(f"\n[SKIP] {tag}/{wtag}: {wname} not found")
                continue
            od = exp / f"test_{tag}_{wtag}"
            od.mkdir(parents=True, exist_ok=True)
            preds = od / "predictions.json"

            sh([sys.executable, SCRIPTS / "predict_rtdetr.py",
                "--config", cfg_path, "--checkpoint", ckpt,
                "--coco_path", view, "--output", preds,
                "--min-score", a.min_score], cwd=SCRIPTS, dry=a.dry_run)

            sh([sys.executable, SCRIPTS / "evaluate_predictions.py",
                "--predictions", preds,
                "--ground_truth", view / "annotations/instances_val2017.json",
                "--output_dir", od, "--model", "rtdetr"],
               cwd=SCRIPTS, dry=a.dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true",
                    help="four proportions by ten seeds")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exp-root", default=EXP_ROOT)
    a = ap.parse_args()

    check_isolation(a.exp_root)
    Path(a.exp_root).mkdir(parents=True, exist_ok=True)

    if a.all:
        jobs = [(v, s) for v in VARIANTS for s in SEEDS]
    elif a.variant and a.seed is not None:
        jobs = [(a.variant, a.seed)]
    else:
        sys.exit("either --all or both --variant and --seed are required")

    print(f"output root : {a.exp_root}")
    print(f"runs        : {len(jobs)}")
    for v, s in jobs:
        print(f"   {v}  seed={s}")

    for v, s in jobs:
        run_one(v, s, a)

    print(f"\n{'=' * 70}\n FINISHED  ({len(jobs)} runs)\n{'=' * 70}")


main()
