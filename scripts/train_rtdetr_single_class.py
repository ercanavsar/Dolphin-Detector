"""
===============================================================================
RT-DETRv2 benchmark runner, single-class (dolphin-only) setup
===============================================================================
Runs the whole sequence for one scenario:
  1. generates the training configuration and fine-tunes with tools/train.py
  2. generates a configuration for each of the two test sets
  3. produces predictions with the best and last checkpoints
     (predict_rtdetr.py)
  4. converts those predictions into the shared metrics
     (evaluate_predictions.py)

Output: experiments/<S>/rtdetr_single_class/test_{LOCAL,EXT}_{best,last}/

Usage:
  python train_rtdetr_single_class.py --scenario S1 --epochs 100 --batch_size 6
  python train_rtdetr_single_class.py --scenario S1 --epochs 100 --batch_size 6 --eval-only
===============================================================================
"""
import argparse, subprocess, sys
from pathlib import Path
import yaml

SCRIPTS = Path(__file__).resolve().parent

# Set these to the local installation.
RT_ROOT = Path("/path/to/RT-DETR/rtdetrv2_pytorch")
VIEWS = Path("/path/to/dolphin_study/coco_views_single_class")
TRAIN_CFG = Path("/path/to/Dolphin-Detector/configs/train_config.yaml")

RTCFG = RT_ROOT / "configs/rtdetrv2"
PRETRAIN = RT_ROOT / "weights/rtdetrv2_r50vd_6x_coco_ema.pth"


def sh(cmd, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        sys.exit(f"ERROR: command failed (exit {r.returncode})")


def make_config(scenario, split, epochs, batch_size):
    cmd = [sys.executable, SCRIPTS/"make_rtdetr_config.py",
           "--scenario", scenario, "--split", split]
    if epochs:
        cmd += ["--epochs", epochs]
    if batch_size:
        cmd += ["--batch_size", batch_size]
    sh(cmd, cwd=SCRIPTS)
    name = f"{scenario}_single_class" + (f"_{split}" if split != "train" else "")
    return RTCFG / f"{name}.yml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training and evaluate the existing checkpoints")
    ap.add_argument("--min-score", type=float, default=0.001)
    a = ap.parse_args()

    g = yaml.safe_load(open(TRAIN_CFG))["global"]
    exp = Path(g["output_dir"]) / a.scenario / "rtdetr_single_class"

    if not PRETRAIN.is_file():
        sys.exit(f"ERROR: COCO checkpoint not found -> {PRETRAIN}")

    # --- 1. training ---
    train_cfg = make_config(a.scenario, "train", a.epochs, a.batch_size)
    if not a.eval_only:
        print(f"\n{'='*60}\n TRAINING  {a.scenario}\n{'='*60}")
        sh([sys.executable, "tools/train.py",
            "-c", train_cfg.relative_to(RT_ROOT),
            "-t", PRETRAIN.relative_to(RT_ROOT),
            "--seed", g["seed"]], cwd=RT_ROOT)

    # --- 2-4. evaluation ---
    for tag, split in [("LOCAL", "test_local"), ("EXT", "test_ext")]:
        view = VIEWS / f"{a.scenario}_TEST_{tag}"
        if not view.is_dir():
            print(f"\n[SKIP] {tag}: view not found ({view.name})")
            continue

        test_cfg = make_config(a.scenario, split, a.epochs, a.batch_size)

        for wtag, wname in [("best", "best.pth"), ("last", "last.pth")]:
            ckpt = exp / wname
            if not ckpt.is_file():
                print(f"\n[SKIP] {tag}/{wtag}: {ckpt.name} not found")
                continue

            print(f"\n{'='*60}\n TEST {tag} / {wtag}\n{'='*60}")
            out_dir = exp / f"test_{tag}_{wtag}"
            preds = out_dir / "predictions.json"
            out_dir.mkdir(parents=True, exist_ok=True)

            sh([sys.executable, SCRIPTS/"predict_rtdetr.py",
                "--config", test_cfg,
                "--checkpoint", ckpt,
                "--coco_path", view,
                "--output", preds,
                "--min-score", a.min_score], cwd=SCRIPTS)

            sh([sys.executable, SCRIPTS/"evaluate_predictions.py",
                "--predictions", preds,
                "--ground_truth", view/"annotations/instances_val2017.json",
                "--output_dir", out_dir,
                "--model", "rtdetrv2"], cwd=SCRIPTS)

    print(f"\n{'='*60}\n FINISHED: {exp}\n{'='*60}")


main()
