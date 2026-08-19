# FBDNet

**Frequency-Guided Boundary–Distance Learning for Agricultural Field Instance Segmentation in High-Resolution Remote Sensing Imagery**

FBDNet is a frequency-guided, boundary-constrained, and distance-aware framework for agricultural field instance segmentation in high-resolution remote sensing imagery. The method combines frequency-guided self-supervised representation learning, boundary-guided multi-branch prediction, and adaptive instance decoding to improve the separation of adjacent field instances under weak-boundary and strong intra-field texture conditions.

## Associated Paper

**Title:** FBDNet: Frequency-Guided Boundary–Distance Learning for Agricultural Field Instance Segmentation in High-Resolution Remote Sensing Imagery
**Authors:** Zhongshui Qu, Yanjie Liu, Jinyan Shi, Jun Zhou, Bo Ding
**Target journal:** *Computers & Geosciences*
**Corresponding author:** Bo Ding
**E-mail:** [dingbo@hrbust.edu.cn](mailto:dingbo@hrbust.edu.cn)
**Status:** Manuscript prepared for submission.

> The paper DOI and final bibliographic information will be added after publication.

## Features

* **FGE-MAE pretraining** with RGB reconstruction and frequency-domain structure reconstruction using Gabor structural targets.
* **Gabor Target Generator** with multi-scale and multi-orientation structural responses.
* **Boundary-Guided Foreground Refinement (BGFR)** for boundary-local refinement of semantic predictions.
* **Three-branch prediction head** for semantic, boundary, and normalized distance responses.
* **Boundary-Weighted Binary Loss (BWBL)** with skeleton-enhanced and neighborhood-silenced boundary supervision.
* **Center-weighted distance regression** for learning instance-centered distance responses.
* **Adaptive instance decoding** based on semantic foreground masking, boundary-constrained separation, region-wise distance normalization, watershed segmentation, and shallow-valley merging.
* Reference implementations of evaluation utilities for semantic, boundary, and instance-level analysis.
* A runnable synthetic demo for validating the adaptive instance decoding module.

## Repository Layout

```text
FBDNet/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── fbdnet_m5.py
├── models/
│   ├── __init__.py
│   ├── fbdnet.py
│   ├── bgfr.py
│   └── losses.py
├── pretraining/
│   ├── __init__.py
│   └── fge_mae.py
├── decoding/
│   ├── __init__.py
│   └── adaptive_instance_decoding.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── demo/
│   └── run_demo.py
├── train.py
└── test.py
```

## Current Repository Scope

This repository currently provides a lightweight reference implementation of the principal FBDNet components described in the manuscript.

The following components are implemented directly in the current release:

* Gabor-based structural target generation and the dual reconstruction loss used by FGE-MAE.
* UPer-style feature fusion and the semantic, boundary, and distance prediction heads.
* BGFR boundary-guided semantic-logit refinement.
* BWBL and the implemented semantic/distance losses.
* Adaptive instance decoding.
* Semantic/boundary metrics and basic instance matching utilities.
* A synthetic decoding demo.

## Environment

### Recommended Software Environment

The repository is prepared for the following environment:

* Linux
* Python 3.10
* CUDA 12.1
* PyTorch 2.1.0
* torchvision 0.16.0
* MMCV 2.1.0
* MMEngine 0.10.7
* MMSegmentation 1.2.2
* MMSelfSup 1.0.0

Additional Python dependencies are listed in [`requirements.txt`](requirements.txt), including NumPy, OpenCV, SciPy, scikit-image, matplotlib, tqdm, einops, and timm.

### Hardware Used in the Paper

The experiments reported in the manuscript were conducted using two NVIDIA RTX 3090 GPUs, each with 24 GB of memory.

The manuscript reports the following main training settings:

* Self-supervised pretraining input size: `256 × 256`
* Self-supervised pretraining global batch size: `32`
* Supervised fine-tuning input size: `512 × 512`
* Supervised fine-tuning global batch size: `8`

## Installation

Create and activate a Conda environment:

```bash
conda create -n fbdnet python=3.10 -y
conda activate fbdnet
```

Install PyTorch and torchvision:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

Install OpenMIM and the OpenMMLab dependencies:

```bash
pip install -U openmim
mim install mmcv==2.1.0
pip install mmengine==0.10.7 mmsegmentation==1.2.2 mmselfsup==1.0.0
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data

The manuscript uses one in-house dataset and two public datasets.

### Qixing Farm Dataset

The in-house Qixing Farm dataset consists of high-resolution aerial digital orthophoto map (DOM) imagery and manually delineated agricultural field instance annotations.

The Qixing Farm dataset cannot be publicly released due to data ownership and confidentiality restrictions. Consequently, the original Qixing Farm imagery and instance annotations are **not included in this repository**.

The manuscript uses the Qixing Farm dataset for self-supervised pretraining, supervised fine-tuning, validation, and the primary test experiments.

For a future complete training release, the expected dataset interface should provide at least:

```text
data/fbdnet/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/
```

The exact data loader and annotation conversion pipeline are not included in the current reference release.

### AI4Boundaries

AI4Boundaries is a publicly available dataset for agricultural field-boundary analysis. The paper uses the **1-m aerial orthophoto subset** for external evaluation.

* Dataset record: `https://data.europa.eu/89h/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9`
* Dataset helper package: `https://github.com/waldnerf/ai4boundaries`
* Data description paper: `https://doi.org/10.5194/essd-15-317-2023`

### FarmSeg-VL

FarmSeg-VL is a publicly available farmland segmentation dataset covering multiple agricultural regions and seasons in China.

* Dataset: `https://doi.org/10.5281/zenodo.15860191`
* Data description paper: `https://doi.org/10.5194/essd-17-4835-2025`

## Method Overview

FBDNet contains three main stages.

### 1. Frequency-Guided Dual-Branch Self-Supervised Pretraining

The pretraining stage extends masked image reconstruction with a frequency-domain structure reconstruction objective.

The released `pretraining/fge_mae.py` contains:

* `GaborTargetGenerator`: generates multi-scale, multi-orientation Gabor structural responses.
* `FGEMAELoss`: combines masked RGB reconstruction loss with masked frequency-structure reconstruction loss.

The default example configuration uses:

```text
Patch size: 16
Frequency reconstruction loss weight: 0.5
Gabor scales: (3, 5, 7)
Number of orientations: 4
Gabor target normalization: enabled
```

### 2. Boundary-Guided Three-Branch Supervised Fine-Tuning

The supervised stage predicts three complementary outputs:

1. semantic foreground/background logits,
2. boundary logits,
3. a normalized distance response map.

The principal implementation is provided in:

```text
models/fbdnet.py
models/bgfr.py
models/losses.py
```

#### Boundary-Guided Foreground Refinement

`BoundaryGuidedFeatureResidual` predicts a boundary response from a shallow feature and uses the boundary probability to gate a residual correction in semantic-logit space:

```text
semantic logits = base logits + boundary-gated residual
```

This implementation corresponds to the boundary-guided semantic refinement used by FBDNet.

#### Boundary-Weighted Binary Loss

`BoundaryWeightedBinaryLoss` gives larger weights to one-pixel boundary-skeleton targets, ignores a configurable neighborhood around the skeleton, and retains standard weights elsewhere.

The example configuration uses:

```text
Skeleton weight: 5.0
Boundary buffer radius: 3 pixels
```

#### Distance Prediction

The distance branch predicts a normalized distance response and is optimized with a center-weighted Smooth L1 loss over foreground pixels.

The example configuration uses:

```text
Distance loss weight: 0.20
Center-weight coefficient: 3.0
Smooth L1 beta: 0.1
```

### 3. Adaptive Instance Decoding

The adaptive instance decoder converts semantic, boundary, and distance outputs into an integer instance map.

The implementation is provided in:

```text
decoding/adaptive_instance_decoding.py
```

The current implementation performs:

1. semantic foreground thresholding,
2. suppression of high-confidence predicted boundary pixels in the semantic foreground,
3. semantic-mask cleaning using morphological opening, small-island removal, and small-hole filling,
4. connected-region-specific normalization of the predicted distance response,
5. local-maximum marker generation,
6. watershed segmentation on the negative smoothed distance map,
7. shallow-watershed merging using peak-to-saddle separation.

The default decoding parameters in `configs/fbdnet_m5.py` are:

| Parameter         | Value |
| ----------------- | ----: |
| `semantic_thr`    |  0.50 |
| `boundary_thr`    |  0.30 |
| `boundary_dilate` |     1 |
| `open_radius`     |     5 |
| `min_island_area` |  2000 |
| `max_hole_area`   | 20000 |
| `sigma`           |   1.5 |
| `peak_thr`        |  0.30 |
| `min_dist`        |    15 |
| `max_markers`     |    50 |
| `min_area_multi`  | 50000 |
| `min_sep`         |  0.20 |

## Configuration

The main example configuration is:

```text
configs/fbdnet_m5.py
```

It records the model-head, pretraining, loss, and instance-decoding settings provided in the current reference release.

Important values currently defined in the configuration include:

```text
Model name: FBDNet-M5
Backbone patch size: 16
Backbone embedding dimension: 768
Backbone depth: 12
Backbone attention heads: 12
Backbone output indices: (2, 5, 8, 11)
Decoder channels: 256
Number of semantic classes: 2
Boundary-band width: 3
Boundary-band loss weight: 0.10
BWBL branch loss weight: 0.30
Distance loss weight: 0.20
Skeleton weight: 5.0
Boundary buffer radius: 3
```

Users should treat `configs/fbdnet_m5.py` as the source of truth for the parameters implemented by this repository snapshot.

## Usage

### Configuration Check / Training Interface

The current training entry point can be used to verify that the configuration is loaded correctly:

```bash
python train.py --config configs/fbdnet_m5.py
```

Optional arguments:

```bash
python train.py \
    --config configs/fbdnet_m5.py \
    --data-root data/fbdnet \
    --work-dir work_dirs/fbdnet_m5
```

At present, `train.py` prints the selected configuration, data root, and work directory. It does not yet execute the full optimization pipeline.

### Inference Interface

The current test entry point exposes the intended inference arguments:

```bash
python test.py \
    --config configs/fbdnet_m5.py \
    --checkpoint checkpoints/fbdnet.pth \
    --image demo/sample.png \
    --out-dir outputs/fbdnet
```

At present, `test.py` reports the requested paths and intended semantic/boundary/distance/adaptive-decoding workflow. It does not yet load a checkpoint or perform model inference.

### Synthetic Adaptive-Decoding Demo

The runnable demo validates the adaptive instance decoding implementation using synthetic semantic, distance, and boundary probability maps:

```bash
python demo/run_demo.py
```

With the current release, the expected console output is:

```text
Synthetic instances: 3
Instance map shape: (256, 256)
```

This demo does not require the Qixing Farm dataset or a trained FBDNet checkpoint.

## Evaluation

Evaluation utilities are provided in:

```text
evaluation/metrics.py
```

The current implementation includes:

* pairwise IoU computation between predicted and ground-truth instances,
* greedy instance matching and F1 computation at a specified IoU threshold,
* binary semantic precision, recall, F1, foreground IoU, background IoU, mIoU, and overall accuracy,
* relaxed boundary IoU and boundary F1 computed from rasterized foreground boundaries.

## Model Weights

Trained model weights are **not included in the current repository release**.

The source modules, configuration file, adaptive decoding implementation, and synthetic test case are provided for inspection and methodological reuse. If model weights are released in the future, the download location will be added here.

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

## Citation

If you find this repository useful, please cite the associated manuscript. Bibliographic information will be updated after publication.

A provisional citation is:

```bibtex
@unpublished{FBDNet2026,
  author = {Qu, Zhongshui and Liu, Yanjie and Shi, Jinyan and Zhou, Jun and Ding, Bo},
  title  = {FBDNet: Frequency-Guided Boundary--Distance Learning for Agricultural Field Instance Segmentation in High-Resolution Remote Sensing Imagery},
  note   = {Manuscript prepared for submission to Computers \& Geosciences},
  year   = {2026}
}
```

## Contact

For questions regarding the implementation or the associated manuscript, please contact:

**Bo Ding**
School of Computer Science and Technology
Harbin University of Science and Technology
E-mail: [dingbo@hrbust.edu.cn](mailto:dingbo@hrbust.edu.cn)

For code-specific issues after the repository is made public, please use the repository issue tracker when possible.
