#!/usr/bin/env python3
"""
===============================================================================
Bridge from train_config.yaml to the main.py of DINO
===============================================================================
Reports how each setting in the shared configuration is handled by DINO:
  [NATIVE]  the DINO default already matches the configuration
  [MAPPED]  translated into a DINO argument
  [SKIP]    no counterpart in DINO, not applied
The report is also written to mapping_report.txt in the output directory.
===============================================================================
"""
import argparse, json, subprocess, sys, shlex
from pathlib import Path
import yaml

# Set these to the local installation.
DINO_ROOT = Path("/path/to/DINO")
TRAIN_CFG = "/path/to/Dolphin-Detector/configs/train_config.yaml"
VIEWS_ROOT = "/path/to/dolphin_study/coco_views_single_class"

REPORT = []


def rec(tag, key, val, note=""):
    REPORT.append((tag, key, str(val), note))


def detect_num_classes(view_dir):
    """Work out the required num_classes from the annotation files."""
    mx = -1
    for split in ["train2017", "val2017"]:
        p = view_dir / "annotations" / f"instances_{split}.json"
        d = json.load(open(p))
        mx = max(mx, max(c["id"] for c in d["categories"]))
    return mx + 1


def build(cfg, scenario, args, views_root):
    g = cfg["global"]
    aug = cfg["augmentations"]
    m = cfg["models"]["dino"]

    view = views_root / scenario
    if not view.is_dir():
        sys.exit(f"ERROR: view not found -> {view}\n"
                 f"  run first: python make_coco_view.py --scenario {scenario}")

    nc = detect_num_classes(view)
    rec("MAPPED", "num_classes", nc, "taken from the annotation files")

    epochs = args.epochs or g["max_epochs"]
    if args.epochs:
        rec("MAPPED", "epochs", epochs,
            f"overridden on the command line (YAML: {g['max_epochs']})")
    else:
        rec("MAPPED", "max_epochs -> epochs", epochs)

    bs = args.batch_size or g["batch_size"]
    note = "" if bs <= 4 else "may not fit on a 40 GB A100"
    rec("MAPPED", "batch_size", bs, note)

    lr_drop = (args.lr_drop if args.lr_drop is not None
               else max(1, int(round(epochs * 0.75))))
    rec("MAPPED", "lr_drop", lr_drop,
        "not in the YAML; set to epochs*0.75, as DINO uses step decay")

    # --- mapped directly ---
    rec("MAPPED", "lr0 -> lr", m["lr0"])
    rec("MAPPED", "lr_backbone", m["lr_backbone"])
    rec("MAPPED", "weight_decay", m["weight_decay"])
    rec("MAPPED", "workers -> --num_workers", g["workers"])
    rec("MAPPED", "seed -> --seed", g["seed"])

    # --- already identical ---
    rec("NATIVE", "optimizer", m["optimizer"], "DINO already uses AdamW")
    rec("NATIVE", "fliplr", aug["fliplr"], "DINO RandomHorizontalFlip p=0.5")
    rec("NATIVE", "flipud", aug["flipud"], "DINO does not flip vertically")
    rec("NATIVE", "mosaic / mixup", 0.0,
        "not available in DINO, so already off")

    # --- no counterpart ---
    rec("SKIP", "img_size", g["img_size"],
        "DINO rescales the shorter side to 480-800, longer side up to 1333")
    rec("SKIP", "warmup_epochs", m["warmup_epochs"], "DINO uses no warmup")
    rec("SKIP", "lrf", m.get("lrf", "-"),
        "DINO uses step decay rather than a final-rate schedule")
    for k in ["degrees", "translate", "scale", "hsv_h", "hsv_s", "hsv_v"]:
        rec("SKIP", k, aug[k], "no counterpart in the DINO transforms")
    rec("SKIP", "patience", g["patience"],
        "no early stopping in DINO; checkpoint_best_regular.pth is saved")

    rec("SKIP", "dn_label_coef / dn_bbox_coef / dn_scalar", "-",
        "mentioned in the official README but never read in this code base")
    rec("NATIVE", "cls/bbox/giou_loss_coef", "1.0 / 5.0 / 2.0",
        "configuration default; the denoising losses use the same weights")
    rec("NATIVE", "dn_number", 100,
        "configuration default, the number of denoising queries")

    out_dir = (Path(args.output_dir) if args.output_dir else
               Path(g["output_dir"]) / scenario / f"dino{args.tag}")

    opts = [
        f"epochs={epochs}", f"batch_size={bs}",
        f"num_classes={nc}", f"dn_labelbook_size={nc}",
        f"lr={m['lr0']}", f"lr_backbone={m['lr_backbone']}",
        f"weight_decay={m['weight_decay']}", f"lr_drop={lr_drop}",
        "embed_init_tgt=TRUE", "use_ema=False", "dn_box_noise_scale=1.0",
    ]

    cmd = [
        sys.executable, "main.py",
        "-c", args.config,
        "--coco_path", str(view),
        "--output_dir", str(out_dir),
        "--num_workers", str(g["workers"]),
        "--seed", str(g["seed"]),
    ]
    if args.eval:
        cmd += ["--eval", "--resume", args.resume]
    else:
        cmd += ["--pretrain_model_path", args.pretrain,
                "--finetune_ignore", "label_enc.weight", "class_embed"]
    cmd += ["--options"] + opts

    return cmd, out_dir


def print_report(out_dir):
    lines = []
    for tag in ["NATIVE", "MAPPED", "SKIP"]:
        rows = [r for r in REPORT if r[0] == tag]
        if not rows:
            continue
        lines.append(f"\n--- {tag} ---")
        for _, k, v, note in rows:
            lines.append(f"  {k:26s} = {v:<12s} {note}")
    txt = "\n".join(lines)
    print(txt)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mapping_report.txt").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--yaml", default=TRAIN_CFG)
    ap.add_argument("--config", default="config/DINO/DINO_4scale.py")
    ap.add_argument("--pretrain",
                    default="checkpoints/checkpoint0011_4scale.pth")
    ap.add_argument("--epochs", type=int,
                    help="overrides max_epochs from the YAML")
    ap.add_argument("--batch_size", type=int,
                    help="overrides batch_size from the YAML")
    ap.add_argument("--lr_drop", type=int)
    ap.add_argument("--output_dir")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--resume", help="checkpoint to use with --eval")
    ap.add_argument("--run", action="store_true",
                    help="actually run the command; "
                         "without it the command is only printed")
    ap.add_argument("--views-root", default=VIEWS_ROOT)
    args = ap.parse_args()

    if args.eval and not args.resume:
        sys.exit("ERROR: --eval requires --resume")

    cfg = yaml.safe_load(open(args.yaml))
    cmd, out_dir = build(cfg, args.scenario, args, Path(args.views_root))

    print(f"=== mapping from the YAML onto DINO ({args.scenario}) ===")
    print_report(out_dir)
    print(f"\n=== command ===\n{' '.join(shlex.quote(c) for c in cmd)}\n")
    print(f"output_dir: {out_dir}")

    if args.run:
        sys.exit(subprocess.call(cmd, cwd=DINO_ROOT))
    else:
        print("\n(--run not given, nothing was executed)")


main()
