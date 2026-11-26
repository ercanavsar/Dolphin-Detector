#!/usr/bin/env python3
"""
===============================================================================
Model-independent evaluation (single-class, dolphin-only setup)
===============================================================================
Input : COCO-format prediction JSON + ground-truth annotation JSON
Output: summary_metrics.json, false_alarms.json,
        fp_vs_confidence.csv, confusion_matrix.csv

The same metric definitions are applied to DINO, YOLO and RT-DETRv2; the only
difference between them is the script that produces the predictions.

Metric definitions:
  mAP / AR            pycocotools (COCO protocol)
  precision/recall/f1 at the threshold that maximises F1, IoU 0.5
  false alarms        every detection in a negative image counts as a false
                      positive; in a positive image, a detection that matches
                      no ground-truth box at IoU > 0.5 counts as one. Both are
                      taken at the same threshold as precision and recall.
  confusion matrix    2x2, conf = 0.25, IoU = 0.45
===============================================================================
"""
import argparse, json, csv, collections
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CONF_TH, CM_IOU_TH, PR_IOU_TH = 0.25, 0.45, 0.5

# Confidence grid for the operating curve. The low end is needed because the
# maximum-F1 threshold of some runs falls below 0.05, and the high end because
# the transformer-based models operate close to 1.0.
CONF_GRID = ([0.005, 0.01, 0.02, 0.03, 0.04]
             + [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
             + [0.96, 0.97, 0.98, 0.99])
EPS = 1e-16


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1])
    ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb-lt, 0, None)
    inter = wh[..., 0]*wh[..., 1]
    return inter/(aa[:, None]+ab[None, :]-inter+EPS)


def load(pred_path, gt_path):
    """Reduce predictions and ground truth to per-image
    (pred_boxes, scores, gt_boxes) records."""
    gt = json.load(open(gt_path))
    folders = {i["id"]: i["file_name"].split("/")[0] for i in gt["images"]}
    gt_by_img = collections.defaultdict(list)
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img[a["image_id"]].append([x, y, x+w, y+h])

    preds = json.load(open(pred_path))
    if isinstance(preds, dict):           # some formats use {"annotations": [...]}
        preds = preds.get("annotations", [])
    pr_by_img = collections.defaultdict(list)
    for p in preds:
        x, y, w, h = p["bbox"]
        pr_by_img[p["image_id"]].append(([x, y, x+w, y+h], p["score"]))

    unknown = set(pr_by_img) - set(folders)
    if unknown:
        raise SystemExit(
            f"ERROR: {len(unknown)} predictions carry an image_id that is not "
            f"in the ground truth. Example: {list(unknown)[:3]}\n"
            f"  Predictions must use the same image_id values as the "
            f"ground-truth file.")

    recs = []
    for iid in sorted(folders):
        pl = pr_by_img.get(iid, [])
        recs.append(dict(
            image_id=iid, folder=folders[iid],
            pb=np.array([b for b, _ in pl]) if pl else np.zeros((0, 4)),
            ps=np.array([s for _, s in pl]) if pl else np.zeros(0),
            gb=np.array(gt_by_img.get(iid, [])) if gt_by_img.get(iid)
               else np.zeros((0, 4))))
    return recs, gt


def coco_metrics(pred_path, gt_path):
    coco = COCO(gt_path)
    preds = json.load(open(pred_path))
    if isinstance(preds, dict):
        preds = preds.get("annotations", [])
    if not preds:
        return {k: 0.0 for k in ["mAP50", "mAP50_95", "AP_small",
                                 "AP_medium", "AP_large", "AR100"]}
    E = COCOeval(coco, coco.loadRes(preds), "bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    return dict(mAP50=float(E.stats[1]), mAP50_95=float(E.stats[0]),
                AP_small=float(E.stats[3]), AP_medium=float(E.stats[4]),
                AP_large=float(E.stats[5]), AR100=float(E.stats[8]))


def max_f1_pr(recs):
    """Precision, recall and F1 at the threshold that maximises F1."""
    px = np.linspace(0, 1, 1000)
    confs, tps, n_gt = [], [], 0
    for r in recs:
        n_gt += len(r["gb"])
        if len(r["pb"]) == 0:
            continue
        o = np.argsort(-r["ps"])
        p, s = r["pb"][o], r["ps"][o]
        tp = np.zeros(len(p))
        if len(r["gb"]):
            iou = box_iou(p, r["gb"]); used = set()
            for i in range(len(p)):
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= PR_IOU_TH and j not in used:
                    tp[i] = 1; used.add(j)
        confs.append(s); tps.append(tp)
    if not confs or n_gt == 0:
        return dict(precision=None, recall=None, f1=None, conf=None)
    confs = np.concatenate(confs); tps = np.concatenate(tps)
    o = np.argsort(-confs); confs, tps = confs[o], tps[o]
    tpc = tps.cumsum(); fpc = (1-tps).cumsum()
    R = np.interp(-px, -confs, tpc/(n_gt+EPS), left=0)
    P = np.interp(-px, -confs, tpc/(tpc+fpc+EPS), left=1)
    f1 = 2*P*R/(P+R+EPS)
    k = int(f1.argmax())
    return dict(precision=float(P[k]), recall=float(R[k]),
                f1=float(f1[k]), conf=float(px[k]))


def recall_at_conf(recs, conf):
    """Recall at a given confidence threshold, matching at IoU 0.5."""
    tp = n_gt = 0
    for r in recs:
        n_gt += len(r["gb"])
        if len(r["gb"]) == 0 or len(r["ps"]) == 0:
            continue
        keep = r["ps"] > conf
        p, s = r["pb"][keep], r["ps"][keep]
        if len(p) == 0:
            continue
        p = p[np.argsort(-s)]
        iou = box_iou(p, r["gb"])
        used = set()
        for i in range(len(p)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= PR_IOU_TH and j not in used:
                tp += 1
                used.add(j)
    return tp / (n_gt + EPS)


def _fmt(d):
    return dict(images=d["images"], negative_images=d["neg"],
                total_fp=d["fp"],
                fp_per_image=d["fp"]/max(d["images"], 1),
                pct_images_with_fp=100*d["img_fp"]/max(d["images"], 1),
                fp_per_negative_image=d["neg_fp"]/d["neg"] if d["neg"] else None,
                pct_negative_images_with_fp=100*d["neg_img_fp"]/d["neg"]
                if d["neg"] else None)


def false_alarms(recs, conf):
    """False-alarm rates, reported per source subset and in total."""
    pf = collections.defaultdict(lambda: dict(
        images=0, neg=0, fp=0, img_fp=0, neg_fp=0, neg_img_fp=0))
    for r in recs:
        d = pf[r["folder"]]; d["images"] += 1
        p = r["pb"][r["ps"] > conf] if len(r["ps"]) else r["pb"]
        is_neg = len(r["gb"]) == 0
        if is_neg:
            d["neg"] += 1; fp = len(p)
        else:
            fp = int((box_iou(p, r["gb"]).max(1) < PR_IOU_TH).sum()) if len(p) else 0
        d["fp"] += fp
        if fp:
            d["img_fp"] += 1
        if is_neg:
            d["neg_fp"] += fp
            if fp:
                d["neg_img_fp"] += 1
    out, tot = {}, collections.Counter()
    for f, d in sorted(pf.items()):
        for k, v in d.items():
            tot[k] += v
        out[f] = _fmt(d)
    out["_ALL_"] = _fmt(tot)
    return out


def confusion_2x2(recs, conf):
    cm = np.zeros((2, 2), dtype=int)
    for r in recs:
        p = r["pb"][r["ps"] > conf] if len(r["ps"]) else r["pb"]
        gb = r["gb"]
        mg = mp = set()
        if len(gb) and len(p):
            iou = box_iou(gb, p)
            gi, pi = np.where(iou > CM_IOU_TH)
            if len(gi):
                m = np.stack([gi, pi, iou[gi, pi]], 1)
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
                mg = set(m[:, 0].astype(int)); mp = set(m[:, 1].astype(int))
        cm[0, 0] += len(mg)
        cm[0, 1] += len(gb) - len(mg)
        cm[1, 0] += len(p) - len(mp)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="predictions in COCO format")
    ap.add_argument("--ground_truth", required=True,
                    help="ground-truth annotation file")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="",
                    help="model name, written into the report")
    a = ap.parse_args()

    recs, gt = load(a.predictions, a.ground_truth)

    det = coco_metrics(a.predictions, a.ground_truth)
    det = {k: (None if v == -1.0 else v) for k, v in det.items()}

    prf = max_f1_pr(recs)

    # False alarms are reported at the same operating point as precision and
    # recall, since the detectors assign scores on different scales.
    fa_conf = prf["conf"] if prf["conf"] is not None else CONF_TH
    fa = false_alarms(recs, fa_conf)
    cm = confusion_2x2(recs, CONF_TH)

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    summary = {
        "detection": {**det,
                      "precision_at_max_f1": prf["precision"],
                      "recall_at_max_f1": prf["recall"],
                      "f1_at_max_f1": prf["f1"]},
        "_meta": {"model": a.model, "class": "dolphin",
                  "n_images": len(recs),
                  "n_gt_boxes": sum(len(r["gb"]) for r in recs),
                  "conf_at_max_f1": prf["conf"],
                  "pr_iou": PR_IOU_TH, "fp_conf": fa_conf,
                  "cm_conf": CONF_TH, "cm_iou": CM_IOU_TH,
                  "predictions": str(a.predictions),
                  "ground_truth": str(a.ground_truth),
                  "note": "single-class setup; non-dolphin objects are "
                          "negative examples, not a separate class"},
    }
    (out/"summary_metrics.json").write_text(json.dumps(summary, indent=4))
    (out/"false_alarms.json").write_text(json.dumps(fa, indent=4))

    with open(out/"confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual\\predicted", "dolphin", "background"])
        w.writerow(["dolphin", int(cm[0, 0]), int(cm[0, 1])])
        w.writerow(["background", int(cm[1, 0]), 0])

    with open(out/"fp_vs_confidence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conf", "recall", "total_fp", "fp_per_image",
                    "pct_images_with_fp", "fp_per_negative_image",
                    "pct_negative_images_with_fp"])
        for c in CONF_GRID:
            d = false_alarms(recs, c)["_ALL_"]
            w.writerow([c, round(recall_at_conf(recs, c), 4),
                        d["total_fp"], round(d["fp_per_image"], 4),
                        round(d["pct_images_with_fp"], 2),
                        round(d["fp_per_negative_image"], 4)
                        if d["fp_per_negative_image"] is not None else "",
                        round(d["pct_negative_images_with_fp"], 2)
                        if d["pct_negative_images_with_fp"] is not None else ""])

    print(f"\n[EXPORT] {out}")
    print(json.dumps(summary["detection"], indent=2))
    print(f"\nfalse alarms (conf = {fa_conf:.3f}):")
    for k, v in fa.items():
        print(f"  {k:24s} fp/img={v['fp_per_image']:.3f}  "
              f"%img={v['pct_images_with_fp']:5.1f}  neg_img={v['negative_images']}")


main()
