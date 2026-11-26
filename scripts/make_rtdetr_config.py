"""
===============================================================================
RT-DETRv2 single-class configuration generator
===============================================================================
Writes one .yml file per scenario into rtdetrv2_pytorch/configs/rtdetrv2/.
It follows the structure of the repository's own rtdetrv2_r50vd_6x_coco.yml,
keeping all __include__ entries in place and overriding only the fields that
need to change.

Settings applied here:
  num_classes = 2            dolphin takes id 1, id 0 is reserved for background
  remap_mscoco_category      False, required for custom category ids
  backbone lr                separate, 1e-5, as in DINO
  use_ema / use_amp          disabled, as in DINO
  augmentation               left at the RT-DETRv2 default; each model keeps
                             its own pipeline and the differences are reported
  multi-scale collate        left at the default
  early stopping             not used; training runs for the full number of
                             epochs
===============================================================================
"""
import argparse, yaml
from pathlib import Path

# Set these to the local installation.
TRAIN_CFG = Path("/path/to/Dolphin-Detector/configs/train_config.yaml")
RT_ROOT = Path("/path/to/RT-DETR/rtdetrv2_pytorch")
VIEWS = Path("/path/to/dolphin_study/coco_views_single_class")
SCENARIOS = Path("/path/to/dolphin_study/scenarios_single_class")

OUT_CFG = RT_ROOT / "configs/rtdetrv2"
NUM_CLASSES = 2


def build(scenario, epochs, batch_size, split):
    g = yaml.safe_load(open(TRAIN_CFG))
    gl, rt = g["global"], g["models"]["rtdetr"]

    if "lr_backbone" not in rt:
        raise SystemExit("train_config.yaml: models.rtdetr.lr_backbone is missing")

    suffix = {"train": "", "test_local": "_TEST_LOCAL",
              "test_ext": "_TEST_EXT"}[split]
    view = VIEWS / (scenario + suffix)
    if not view.is_dir():
        raise SystemExit(
            f"view not found: {view}\n  run first: python make_coco_view.py "
            f"--scenario {scenario} --scenarios-root {SCENARIOS} "
            f"--out-root {VIEWS}")

    ep = epochs or gl["max_epochs"]
    bs = batch_size or gl["batch_size"]
    out_dir = Path(gl["output_dir"]) / scenario / "rtdetr_single_class"

    cfg = {
        "__include__": [
            "../dataset/coco_detection.yml",
            "../runtime.yml",
            "./include/dataloader.yml",
            "./include/optimizer.yml",
            "./include/rtdetrv2_r50vd.yml",
        ],

        # --- number of classes, which has to be set in three places ---
        "num_classes": NUM_CLASSES,
        "remap_mscoco_category": False,
        "RTDETRTransformerv2": {"num_classes": NUM_CLASSES},
        "Criterionv2": {"num_classes": NUM_CLASSES},

        # --- training length and reproducibility ---
        "seed": gl["seed"],
        "epoches": ep,
        "output_dir": str(out_dir),

        # --- kept consistent with DINO ---
        "use_ema": False,
        "use_amp": False,

        # --- optimiser, preserving the parameter groups of the repository ---
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
            "total_batch_size": bs,
            "num_workers": gl["workers"],
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(view / "val2017"),
                "ann_file": str(view / "annotations/instances_val2017.json"),
            },
            "total_batch_size": bs,
            "num_workers": gl["workers"],
        },
    }

    name = f"{scenario}_single_class" + (f"_{split}" if split != "train" else "")
    out = OUT_CFG / f"{name}.yml"
    yaml.dump(cfg, open(out, "w"), default_flow_style=False, sort_keys=False)

    print(f"[OK] {out}")
    print(f"  epochs={ep}  batch={bs}  lr={rt['lr0']}  "
          f"lr_backbone={rt['lr_backbone']}")
    print(f"  train ann : {cfg['train_dataloader']['dataset']['ann_file']}")
    print(f"  val   ann : {cfg['val_dataloader']['dataset']['ann_file']}")
    print(f"  output    : {out_dir}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--split", default="train",
                    choices=["train", "test_local", "test_ext"],
                    help="use test_* to build a configuration for evaluation")
    a = ap.parse_args()
    build(a.scenario, a.epochs, a.batch_size, a.split)
