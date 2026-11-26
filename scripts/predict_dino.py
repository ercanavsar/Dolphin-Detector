#!/usr/bin/env python3
"""
===============================================================================
DINO inference -> predictions in COCO format
===============================================================================
Takes a checkpoint and a coco_view, runs inference on the val2017 split of
that view, and writes the predictions as the COCO-format JSON that
evaluate_predictions.py expects.

DINO produces 900 queries per image. Those below --min-score are discarded and
at most --max-dets are kept, matching the settings used for the other two
detectors (0.001 and 300).
===============================================================================
"""
import argparse, json, sys, os
from pathlib import Path
import numpy as np
import torch

# Set this to the local clone of the DINO repository.
DINO_ROOT = Path("/path/to/DINO")

sys.path.insert(0, str(DINO_ROOT))
os.chdir(DINO_ROOT)

from util.slconfig import SLConfig
from datasets import build_dataset
import util.misc as utils
from models.registry import MODULE_BUILD_FUNCS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/DINO/DINO_4scale.py")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--coco_path", required=True, help="coco_view directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    view = Path(a.coco_path)
    ann = view / "annotations" / "instances_val2017.json"
    if not ann.is_file():
        sys.exit(f"ERROR: {ann} not found")

    cfg = SLConfig.fromfile(a.config)
    args = argparse.Namespace(**cfg._cfg_dict.to_dict())
    args.dataset_file, args.masks = "coco", False
    args.fix_size = args.strong_aug = False
    args.num_classes = args.dn_labelbook_size = a.num_classes
    args.device, args.distributed = a.device, False
    args.coco_path = str(view)

    meta = json.load(open(ann))
    cat_ids = sorted(c["id"] for c in meta["categories"])

    print(f"checkpoint : {a.checkpoint}")
    print(f"view       : {view}")
    print(f"categories : {[(c['id'], c['name']) for c in meta['categories']]}")

    ds = build_dataset(image_set="val", args=args)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, shuffle=False,
        collate_fn=utils.collate_fn, num_workers=a.num_workers)

    model, _, post = MODULE_BUILD_FUNCS.get(args.modelname)(args)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu",
                                     weights_only=False)["model"])
    model.to(a.device).eval()

    dets, n_img = [], 0
    for samples, targets in loader:
        samples = samples.to(a.device)
        sizes = torch.stack([t["orig_size"] for t in targets], 0).to(a.device)
        results = post["bbox"](model(samples), sizes)
        for t, r in zip(targets, results):
            iid = int(t["image_id"])
            n_img += 1
            b = r["boxes"].cpu().numpy()
            s = r["scores"].cpu().numpy()
            l = r["labels"].cpu().numpy()

            keep = s >= a.min_score
            b, s, l = b[keep], s[keep], l[keep]
            if len(s) > a.max_dets:
                idx = np.argsort(s)[::-1][:a.max_dets]
                b, s, l = b[idx], s[idx], l[idx]

            for bb, ss, ll in zip(b, s, l):
                ll = int(ll)
                # DINO uses the label directly as the category_id
                cid = ll if ll in cat_ids else (
                    cat_ids[ll] if ll < len(cat_ids) else ll)
                dets.append(dict(
                    image_id=iid, category_id=cid,
                    bbox=[float(bb[0]), float(bb[1]),
                          float(bb[2] - bb[0]), float(bb[3] - bb[1])],
                    score=float(ss)))

    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dets, open(out, "w"))
    print(f"\n[OK] {n_img} images, {len(dets)} predictions "
          f"(score >= {a.min_score}, max_det={a.max_dets})")
    print(f"[EXPORT] {out}")


main()
