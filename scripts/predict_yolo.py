#!/usr/bin/env python3
"""
===============================================================================
YOLO inference -> predictions in COCO format
===============================================================================
The save_json output of Ultralytics derives the image_id from the file name,
which does not match the ids used in the COCO annotation files. This script
produces the predictions directly and takes the image_id from the ground
truth, so that the result can be read by evaluate_predictions.py.
===============================================================================
"""
import argparse, json
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--coco_path", required=True, help="coco_view directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    view = Path(a.coco_path)
    ann = view / "annotations" / "instances_val2017.json"
    if not ann.is_file():
        raise SystemExit(f"ERROR: {ann} not found")

    gt = json.load(open(ann))
    cat_ids = sorted(c["id"] for c in gt["categories"])
    # file_name -> image_id
    id_by_name = {i["file_name"]: i["id"] for i in gt["images"]}
    images = [(view / "val2017" / fn, iid) for fn, iid in id_by_name.items()]
    images.sort(key=lambda x: x[1])

    print(f"weights    : {a.weights}")
    print(f"view       : {view}")
    print(f"categories : {[(c['id'], c['name']) for c in gt['categories']]}")
    print(f"images     : {len(images)}")

    model = YOLO(a.weights)
    dets = []
    for i in range(0, len(images), a.batch):
        chunk = images[i:i + a.batch]
        res = model.predict([str(p) for p, _ in chunk], imgsz=a.imgsz,
                            conf=a.min_score, max_det=a.max_dets,
                            device=a.device, verbose=False)
        for (_, iid), r in zip(chunk, res):
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            for bb, ss, cc in zip(b, s, c):
                cid = cat_ids[cc] if cc < len(cat_ids) else int(cc)
                dets.append(dict(
                    image_id=iid, category_id=cid,
                    bbox=[float(bb[0]), float(bb[1]),
                          float(bb[2] - bb[0]), float(bb[3] - bb[1])],
                    score=float(ss)))

    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dets, open(out, "w"))
    print(f"\n[OK] {len(dets)} predictions "
          f"(score >= {a.min_score}, max_det={a.max_dets})")
    print(f"[EXPORT] {out}")


main()
