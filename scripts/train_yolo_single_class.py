"""
===============================================================================
YOLO benchmark runner, single-class (dolphin-only) setup
===============================================================================
Runs the whole sequence for one scenario:
  1. fine-tunes YOLO26m on the scenario's training split
  2. produces predictions with the best and last checkpoints (predict_yolo.py)
  3. converts those predictions into the shared metrics
     (evaluate_predictions.py)

Output: experiments/<S>/yolo_single_class/test_{LOCAL,EXT}_{best,last}/

Usage:
  python train_yolo_single_class.py --scenario S1 --epochs 100 --train-only
  python train_yolo_single_class.py --scenario S1 --eval-only

Note on GPU access: training and evaluation have to run as separate processes.
Ultralytics trains inside this script's own process and does not release the
CUDA context, so the subprocess that produces the predictions cannot reach the
same device when it is held in exclusive mode. Calling the script twice, first
with --train-only and then with --eval-only, avoids this. Running it without
either flag does both in one process and works only where the GPU is not held
exclusively.
===============================================================================
"""
import argparse, subprocess, sys, yaml
from pathlib import Path
from ultralytics import YOLO

SCRIPTS = Path(__file__).resolve().parent

# Set these to the local installation.
CONFIG_PATH = Path("/path/to/Dolphin-Detector/configs/train_config.yaml")
SCEN_SC = Path("/path/to/dolphin_study/scenarios_single_class")
VIEWS_SC = Path("/path/to/dolphin_study/coco_views_single_class")


def sh(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        sys.exit(f"ERROR: command failed (exit {r.returncode})")


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def run(scenario, epochs=None, eval_only=False, train_only=False):
    cfg = load_yaml(CONFIG_PATH)
    g, aug, yc = cfg["global"], cfg["augmentations"], cfg["models"]["yolo"]

    sd = SCEN_SC / scenario
    yaml_local = sd / f"{scenario}_local.yaml"
    if not yaml_local.exists():
        raise FileNotFoundError(f"not found: {yaml_local}")

    project = Path(g["output_dir"]) / scenario
    name = "yolo_single_class"
    exp = project / name
    ep = epochs if epochs is not None else g["max_epochs"]

    print(f"\n{'='*60}\n YOLO single-class | {scenario} | epochs={ep} "
          f"| batch={g['batch_size']} | imgsz={g['img_size']}\n"
          f" output: {exp}\n{'='*60}\n")

    # --- 1. training ---
    if not eval_only:
        model = YOLO(yc["model_name"])
        model.train(
            data=str(yaml_local.resolve()), project=str(project.resolve()),
            name=name, exist_ok=True, seed=g["seed"], device=g["device"],
            workers=g["workers"], epochs=ep,
            patience=0,                       # early stopping disabled
            batch=g["batch_size"], imgsz=g["img_size"],
            optimizer=yc["optimizer"], lr0=yc["lr0"], lrf=yc["lrf"],
            weight_decay=yc["weight_decay"], warmup_epochs=yc["warmup_epochs"],
            mosaic=aug["mosaic"], mixup=aug["mixup"], fliplr=aug["fliplr"],
            flipud=aug["flipud"], degrees=aug["degrees"],
            translate=aug["translate"], scale=aug["scale"],
            hsv_h=aug["hsv_h"], hsv_s=aug["hsv_s"], hsv_v=aug["hsv_v"],
            val=True, save=True, plots=True)

        print(f"\n[OK] training finished: {scenario}")

        # release the GPU before the prediction subprocesses start
        import gc, torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[SKIP] training skipped (--eval-only)")

    if train_only:
        print("\n[SKIP] evaluation skipped (--train-only)")
        return

    # --- 2-3. evaluation ---
    for tag in ["LOCAL", "EXT"]:
        view = VIEWS_SC / f"{scenario}_TEST_{tag}"
        if not view.is_dir():
            print(f"[SKIP] {tag}: view not found")
            continue
        for wtag, wname in [("best", "best.pt"), ("last", "last.pt")]:
            w = exp / "weights" / wname
            if not w.exists():
                continue
            print(f"\n--- TEST {tag} / {wtag} ---")
            od = exp / f"test_{tag}_{wtag}"
            od.mkdir(parents=True, exist_ok=True)
            preds = od / "predictions.json"

            sh([sys.executable, SCRIPTS / "predict_yolo.py",
                "--weights", w, "--coco_path", view, "--output", preds,
                "--imgsz", g["img_size"], "--batch", g["batch_size"],
                "--device", g["device"]])

            sh([sys.executable, SCRIPTS / "evaluate_predictions.py",
                "--predictions", preds,
                "--ground_truth", view / "annotations/instances_val2017.json",
                "--output_dir", od, "--model", "yolo26m"])

    print(f"\n{'='*60}\n FINISHED: {exp}\n{'='*60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training and evaluate the existing checkpoints")
    ap.add_argument("--train-only", action="store_true",
                    help="train and stop, leaving the evaluation to a "
                         "separate call")
    a = ap.parse_args()
    run(a.scenario, a.epochs, a.eval_only, a.train_only)
