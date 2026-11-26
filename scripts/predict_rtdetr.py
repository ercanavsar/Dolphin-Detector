#!/usr/bin/env python3
"""
===============================================================================
RT-DETRv2 inference -> predictions in COCO format
===============================================================================
Takes a checkpoint and a coco_view, runs inference on the val2017 split of
that view, and writes the predictions as the COCO-format JSON that
evaluate_predictions.py expects.

Note: the postprocessor returns num_top_queries (300) predictions per image.
Those below --min-score are not written (default 0.001).
===============================================================================
"""
import argparse, json, sys, os
from pathlib import Path
import torch

# Set this to the local clone of the RT-DETR repository.
RT_ROOT = Path("/path/to/RT-DETR/rtdetrv2_pytorch")

sys.path.insert(0, str(RT_ROOT))
os.chdir(RT_ROOT)

from src.core import YAMLConfig


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="the .yml configuration of the scenario")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--coco_path", required=True, help="coco_view directory")
    ap.add_argument("--output", required=True,
                    help="prediction file to write")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-dets", type=int, default=300)
    a = ap.parse_args()

    view = Path(a.coco_path)
    ann = view / "annotations" / "instances_val2017.json"
    if not ann.is_file():
        sys.exit(f"ERROR: {ann} not found")

    # load the configuration and point its val_dataloader at the target view
    cfg = YAMLConfig(a.config)
    cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = str(view / "val2017")
    cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = str(ann)
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = a.batch_size
    cfg.yaml_cfg["val_dataloader"]["shuffle"] = False

    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = state.get("ema", {}).get("module") if state.get("ema") else None
    sd = sd or state["model"]
    cfg.model.load_state_dict(sd)

    model = cfg.model.to(a.device).eval()
    post = cfg.postprocessor.to(a.device).eval()
    loader = cfg.val_dataloader

    cats = json.load(open(ann))["categories"]
    cat_ids = sorted(c["id"] for c in cats)
    print(f"checkpoint : {a.checkpoint}")
    print(f"view       : {view}")
    print(f"categories : {[(c['id'], c['name']) for c in cats]}")

    dets, n_img = [], 0
    for samples, targets in loader:
        samples = samples.to(a.device)
        sizes = torch.stack([t["orig_size"] for t in targets], 0).to(a.device)
        results = post(model(samples), sizes)
        for t, r in zip(targets, results):
            iid = int(t["image_id"])
            n_img += 1
            b = r["boxes"].cpu().numpy()
            s = r["scores"].cpu().numpy()
            l = r["labels"].cpu().numpy()

            keep = s >= a.min_score
            b, s, l = b[keep], s[keep], l[keep]
            if len(s) > a.max_dets:
                idx = s.argsort()[::-1][:a.max_dets]
                b, s, l = b[idx], s[idx], l[idx]

            for bb, ss, ll in zip(b, s, l):
                # internal label index -> COCO category_id
                cid = cat_ids[int(ll)] if int(ll) < len(cat_ids) else int(ll)
                dets.append(dict(
                    image_id=iid, category_id=cid,
                    bbox=[float(bb[0]), float(bb[1]),
                          float(bb[2]-bb[0]), float(bb[3]-bb[1])],
                    score=float(ss)))

    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dets, open(out, "w"))
    print(f"\n[OK] {n_img} images, {len(dets)} predictions "
          f"(score >= {a.min_score})")
    print(f"[EXPORT] {out}")


main()
