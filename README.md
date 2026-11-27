# Dolphin Detector

Code and trained weights accompanying the manuscript *A cross-source benchmark of
object detectors for underwater dolphin detection in bycatch mitigation*
(under review). Links to the published article and the dataset will be added
here once available.

The study aims to support the development of automated bycatch mitigation in towed
fishing gear.

Three object detectors (YOLO26m, RT-DETRv2 and DINO) are trained on the same
data and evaluated by the same procedure across sixteen scenarios, each using a
different combination of data sources for training. The scenarios are built so
that no recording is split between training and testing, which makes it possible
to measure how far a detector transfers to footage it has not seen.

## Contents

```
configs/
    train_config.yaml    hyperparameters and paths used by the training scripts
    scenarios.yaml       which subsets each scenario uses for training and testing
requirements/            package lists for the three environments
scenarios/               the sixteen scenario splits, in COCO format
scripts/                 data preparation, training, inference and evaluation
weights/                 trained weights for the four strongest scenarios
```

## Setup

Each detector is used through its own reference implementation, which must be
cloned separately:

| Detector | Repository |
| --- | --- |
| YOLO26m | https://github.com/ultralytics/ultralytics |
| RT-DETRv2 | https://github.com/lyuwenyu/RT-DETR |
| DINO | https://github.com/IDEA-Research/DINO |

The three have incompatible dependencies, so a separate virtual environment is
needed for each. Package lists are given in `requirements/`.

```bash
python -m venv venv_yolo   && venv_yolo/bin/pip   install -r requirements/req_ultralytics.txt
python -m venv venv_rtdetr && venv_rtdetr/bin/pip install -r requirements/req_rtdetr_v2.txt
python -m venv venv_dino   && venv_dino/bin/pip   install -r requirements/req_dino_r50.txt
```

DINO requires its CUDA operators to be compiled before first use, as described
in its own documentation.

## Data

### Subsets

The dataset is organised into thirteen subsets, each a separate folder of
images. Six of them were recorded at the Genoa Aquarium and are released with
this work; the remaining seven come from external sources and have to be
obtained from their original providers.

| Folder | Notation | Content | Source |
| --- | --- | --- | --- |
| `dolphin_a` | D_local-A | first dolphin pair | released with this work |
| `dolphin_b` | D_local-B | second dolphin pair | released with this work |
| `other_local` | O_local | other fish, separate basin | released with this work |
| `bg_local_a` | BG_local-A | empty basins 1 and 2 | released with this work |
| `bg_local_b` | BG_local-B | empty basins 2 and 3 | released with this work |
| `bg_other_local` | BG_local-O | empty basin 4 | released with this work |
| `online_dolphin_a` | D_ext-A | dolphins, public videos | video platforms |
| `bg_online_dolphin_a` | BG_ext-A | same videos, no animals | video platforms |
| `online_dolphin_b` | D_ext-B | dolphins, benchmark dataset | NDD20 |
| `online_other_a` | O_ext-A | other species in a trawl | trawl recordings |
| `bg_online_other_a` | BG_ext-O-A | same haul, no animals | trawl recordings |
| `online_other_b` | O_ext-B | other species in a trawl | trawl recordings |
| `bg_online_other_b` | BG_ext-O-B | same haul, no animals | trawl recordings |

The aquarium recordings and their annotations are available at [DOI].

The external material comes from the following sources:

| Subset | Where to obtain it |
| --- | --- |
| `online_dolphin_a`, `bg_online_dolphin_a` | https://vimeo.com/23610098 <br> https://vimeo.com/251212037 <br> https://www.youtube.com/watch?v=bUntcnv1nPY <br> https://www.youtube.com/watch?v=G7L4YzGAvMA <br> https://www.youtube.com/watch?v=pT7BWOLpqF8 <br> https://www.youtube.com/watch?v=WLhaPQvCExs |
| `online_dolphin_b` | https://doi.org/10.25405/data.ncl.c.4982342 |
| `online_other_a`, `online_other_b`, `bg_online_other_a`, `bg_online_other_b` | https://doi.org/10.11583/DTU.21769442 |

Frames were extracted from the video sources at intervals of 0.5 seconds.

### Expected layout

The scripts assume the following structure, with one folder per subset:

```
<data root>/
    dataset/
        images/
            dolphin_a/
            dolphin_b/
            ...
        annotations.json
```

The released material follows this layout, so the aquarium folders can be
copied in directly. The external folders are referenced by name in
`configs/scenarios.yaml` and in the scenario files under `scenarios/`, so
frames obtained from the external sources should be placed in folders of the
same names. A scenario that refers to a folder which is empty or absent cannot
be built.

Annotations are in COCO format with two categories, `dolphin` and `other`.
The single-class setup used in this study is derived from them by
`generate_scenarios_single_class.py`, which discards the `other` annotations
and keeps those frames as negative examples. Annotations for the external
subsets are provided by their original sources.

### Paths

The scripts and `configs/train_config.yaml` were written for one particular machine and 
contain absolute paths, pointing at the data root, the scenario folder and the output folder. 
In `train_config.yaml` these are the `dataset_root`, `scenarios_dir` and `output_dir` entries; 
in the scripts they appear in a small block of constants near the top of each file. 
All of them need to be set to the local setup before anything is run.

## Running a scenario

### Preparing the splits

`scenarios/` already contains the sixteen splits in COCO format, so this step
can be skipped if the external subsets are not being reconstructed. To build
them from the annotations instead:

```bash
python scripts/generate_scenarios_single_class.py
```

This reads `configs/scenarios.yaml` together with the annotation file and
writes, for each scenario, the COCO files for training, validation and the two
test sets, as well as the image lists and YAML files that the frameworks
expect. It also writes a set of YOLO label files in which the annotations of
the `other` category have been removed, so that frames showing other species
are treated as background.

The three detectors then need the splits arranged in two different ways. YOLO
locates a label file by replacing `images` with `labels` in the image path, so
it requires a directory in which the images and the single-class labels sit
side by side. `setup_yolo_sc_tree.py` builds this as a pair of symbolic links
and is run once for all scenarios. The two transformer-based models instead
expect the COCO directory layout, which `make_coco_view.py` produces for one
scenario at a time.

```bash
python scripts/setup_yolo_sc_tree.py             # symlink tree used by YOLO26m
python scripts/make_coco_view.py --scenario S4   # COCO layout used by the DETRs
```

On Windows, creating symbolic links requires either administrator rights or
developer mode to be enabled.


### Training and evaluation

One script per detector handles the whole sequence, from training through
inference to the metrics:

```bash
python scripts/train_yolo_single_class.py --scenario S4 --epochs 100 --train-only
python scripts/train_yolo_single_class.py --scenario S4 --eval-only
python scripts/train_rtdetr_single_class.py --scenario S4 --epochs 100 --batch_size 6
python scripts/train_dino_single_class.py   --scenario S4 --epochs 100 --batch_size 6
```


YOLO is called twice because Ultralytics does not release the GPU after
training, which prevents the prediction step from reaching the same device
where it is held in exclusive mode. The other two run in one call.

Each must be run from the environment belonging to its detector. Results are
written to `<output root>/<scenario>/<model>_single_class/`, with one folder per
test set and checkpoint containing the predictions and the computed metrics.

All metrics come from `evaluate_predictions.py`, which is applied identically to
the three detectors so that differences in the results are not affected by the
evaluation code. The training scripts call it automatically, but it can also be
run on its own against any prediction file in COCO format:

```bash
python scripts/evaluate_predictions.py \
    --predictions <path>/predictions.json \
    --ground_truth <view>/annotations/instances_val2017.json \
    --output_dir <path> \
    --model yolo26m
```

It writes the detection metrics, the false-alarm rates broken down by source
subset, the confidence-threshold curve and a confusion matrix.

### Proportion of negative images

The experiments on how much negative material a training set should contain are
run with:

```bash
python scripts/generate_negratio_variants.py --scenario S2
python scripts/run_negabl_rtdetr.py --variant neg05 --seed 1
```

The first builds training sets with 5, 10, 30 and 50 per cent negative images
while keeping the positive frames fixed; the second trains and evaluates one of
them.

## Weights

The final checkpoints for scenarios S4, S8, S12 and S16, which combine local and
external material of both kinds and give the strongest results on the external
test set. All are from a 100-epoch training run.

| Scenario | YOLO26m | RT-DETRv2 | DINO |
| --- | --- | --- | --- |
| S4 | [link] | [link] | [link] |
| S8 | [link] | [link] | [link] |
| S12 | [link] | [link] | [link] |
| S16 | [link] | [link] | [link] |

## Citation

[to be added on publication]

## Funding

This work was funded by the European Union under the Horizon Europe Programme,
Grant Agreement No. 101135237 (Marine Beacon). Research and Innovation Action |
HORIZON-CL6-2023-BIODIV-01. Views and opinions expressed are however those of
the author(s) only and do not necessarily reflect those of the European Union
or European Research Executive Agency (REA). Neither the European Union nor the
granting authority can be held responsible for them.
