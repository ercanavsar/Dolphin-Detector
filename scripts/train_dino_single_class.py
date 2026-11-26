"""
===============================================================================
DINO benchmark runner, single-class (dolphin-only) setup
===============================================================================
Runs the whole sequence for one scenario:
  1. fine-tunes with train_dino.py, which maps the shared configuration onto
     the arguments DINO expects
  2. produces predictions with the best and last checkpoints (predict_dino.py)
  3. converts those predictions into the shared metrics
     (evaluate_predictions.py)

Output: experiments/<S>/dino_single_class/test_{LOCAL,EXT}_{best,last}/

Usage:
  python train_dino_single_class.py --scenario S1 --epochs 100 --batch_size 6
  python train_dino_single_class.py --scenario S1 --eval-only
===============================================================================
"""
import argparse, subprocess, sys
from pathlib import Path
import yaml

SCRIPTS = Path(__file__).resolve().parent

# Set these to the local installation.
VIEWS = Path("/path/to/dolphin_study/coco_views_single_class")
TRAIN_CFG = Path("/path/to/Dolphin-Detector/configs/train_config.yaml")

NUM_CLASSES = 2
TAG = "_single_class"


def sh(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(SCRIPTS))
    if r.returncode != 0:
        sys.exit(f"ERROR: command failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr_drop", type=int)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training and evaluate the existing checkpoints")
    ap.add_argument("--min-score", type=float, default=0.001)
    a = ap.parse_args()

    g = yaml.safe_load(open(TRAIN_CFG))["global"]
    exp = Path(g["output_dir"]) / a.scenario / f"dino{TAG}"

    if not (VIEWS / a.scenario).is_dir():
        sys.exit(f"ERROR: view not found -> {VIEWS / a.scenario}")

    # --- 1. training ---
    if not a.eval_only:
        print(f"\n{'='*60}\n TRAINING  {a.scenario}\n{'='*60}")
        cmd = [sys.executable, SCRIPTS / "train_dino.py",
               "--scenario", a.scenario, "--yaml", TRAIN_CFG,
               "--views-root", VIEWS, "--tag", TAG, "--run"]
        if a.epochs:
            cmd += ["--epochs", a.epochs]
        if a.batch_size:
            cmd += ["--batch_size", a.batch_size]
        if a.lr_drop:
            cmd += ["--lr_drop", a.lr_drop]
        sh(cmd)

    # --- 2-3. evaluation ---
    for tag in ["LOCAL", "EXT"]:
        view = VIEWS / f"{a.scenario}_TEST_{tag}"
        if not view.is_dir():
            print(f"\n[SKIP] {tag}: view not found ({view.name})")
            continue

        for wtag, wname in [("best", "checkpoint_best_regular.pth"),
                            ("last", "checkpoint.pth")]:
            ckpt = exp / wname
            if not ckpt.is_file():
                print(f"\n[SKIP] {tag}/{wtag}: {wname} not found")
                continue

            print(f"\n{'='*60}\n TEST {tag} / {wtag}\n{'='*60}")
            od = exp / f"test_{tag}_{wtag}"
            od.mkdir(parents=True, exist_ok=True)
            preds = od / "predictions.json"

            sh([sys.executable, SCRIPTS / "predict_dino.py",
                "--checkpoint", ckpt, "--coco_path", view,
                "--output", preds, "--num_classes", NUM_CLASSES,
                "--min-score", a.min_score])

            sh([sys.executable, SCRIPTS / "evaluate_predictions.py",
                "--predictions", preds,
                "--ground_truth", view / "annotations/instances_val2017.json",
                "--output_dir", od, "--model", "dino_r50"])

    print(f"\n{'='*60}\n FINISHED: {exp}\n{'='*60}")


main()
