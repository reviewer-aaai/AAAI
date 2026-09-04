# SU(2) × Sim(2) equivariant networks on MNIST — reproduction package

Anonymous artifact for double-blind review.

This repository contains the pre-trained weights and a reviewer notebook that
replays the entire evaluation protocol without retraining anything. Weights are
distributed as self-contained TorchScript archives, so the notebook runs with no
architecture code and no geometric dependency.

**The reviewer notebook is committed with its outputs**, and every figure
below is reproduced by running it. You can verify the paper's claims by reading
this page, without executing a single cell.

---

## 1. Quick start

### Option A — read only (0 minutes)

Open [`Reviewer_Repro_SU2xSim2_MNIST_V10.ipynb`](Reviewer_Repro_SU2xSim2_MNIST_V10.ipynb).
All cells are pre-executed. The same content is available as standalone files
under [`results/`](results/).

### Option B — re-run the evaluation

Download this repository as a zip. Open the reviewer notebook in Colab with a
GPU runtime, upload the zip to the session using the folder icon in the left
sidebar, and run all. The setup cell extracts it and loads the weights.
Nothing needs to be installed.

(There is no Colab badge and no `git clone` step: both would reveal the
repository owner during double-blind review. They will be restored on
acceptance.)

| `RUN_MODE` | Test set | Sweep resolution | Wall clock (T4) |
|---|---|---|---|
| `"quick"` | 2,000 images | reduced | ≈ 5 min |
| `"full"` | 10,000 images | as in the paper | ≈ 35–50 min |

The committed outputs correspond to `RUN_MODE = "full"`.

Nothing needs to be installed: `torch`, `torchvision`, `numpy`, `pandas`,
`matplotlib` and `tqdm` all ship with Colab, and no geometric library is
required.

---

## 2. Results at a glance

### 2.1 Robustness sweeps

Accuracy as a function of transformation amplitude, six families. This is the
paper's central result.

![Robustness curves](results/robustness_curves.png)

| Model | Acc, 10 classes | Acc @ zero amplitude | ΔΦ (Möbius) | Δψ₁ (planar rot.) | Scale range | Projective (μ ± σ) | ΔCombo | SU(2) Haar (μ ± σ) | SU(2) Haar min |
|---|---|---|---|---|---|---|---|---|---|
| Classic ResNet | 99.59 | 99.4 | 92.5 | 78.9 | 72.9 | 35.4 ± 25.2 | 95.6 | 13.1 ± 17.1 | 2.9 |
| Sim(2) only | 99.34 | 99.6 | 91.1 | **19.0** | 74.7 | **73.9 ± 14.0** | 93.2 | 26.1 ± 25.9 | 4.8 |
| SU(2) stereographic | 94.53 | 95.1 | **53.0** | **0.0** | 82.2 | 59.0 ± 26.9 | **53.0** | **49.9 ± 14.0** | **36.4** |
| Bi-LogPolar | 98.57 | 98.6 | 88.6 | 31.2 | 82.4 | 54.0 ± 20.5 | 94.2 | 15.4 ± 18.1 | 4.1 |

All values are percentages. Δ is the accuracy drop between zero and maximum
amplitude, so **lower Δ means more invariant**. The first column covers all ten
classes; every other column comes from the sweeps, which exclude classes 1 and
9 (see [Section 7](#7-protocol-notes)). The two are therefore not directly
comparable, and both fluctuate by up to ±0.25 points between runs because the
evaluation transform applies a random translation.

The four models are calibrated to the same parameter budget (500k, via
`find_model_width`), so the comparison is capacity-matched.

**Reading the table.** The SU(2) stereographic model is exactly invariant to
planar rotation (Δψ₁ = 0.0, a flat green line in the ψ₁ panel above), and it is
the only model that survives Haar-distributed SU(2) transformations at a useful
rate: mean 49.9% with a worst case of 36.4%, against 13.1% mean and 2.9% worst
case for the classical baseline. Under composed φ+ψ₁ Möbius transformations its
drop is roughly half that of every other model.

Three costs are equally visible and are reported here rather than buried. Clean
accuracy is 4.4 points below the baselines. Scale robustness is the worst of
the four (range 82.2), the model degrading sharply below 0.7× and above 1.5×.
Under random projective homographies the Sim(2) model is ahead (73.9% versus
59.0%), which is expected since those transformations are not in SU(2).

### 2.2 Architectural equivariance

Cosine error between the global descriptor of `x` and of `T·x`, at random
initialisation. Equivariance here is a property of the architecture alone, not
of training.

![Equivariance at initialisation](results/equivariance_random.png)

Random versus trained weights, same measurement:

![Equivariance, random vs trained](results/equivariance_comparison.png)

At initialisation the geometric models sit well under the 15% threshold on
every family while the classical baseline exceeds it on rotation and scale.
After training, the errors rise for all four models, indicating that
optimisation on MNIST partially trades feature-level equivariance for class
discriminability. The decision-level invariance measured in Section 2.1 is
nevertheless preserved.

### 2.3 Transformation families

Control panels showing that the tested amplitudes remain semantically valid.

**Möbius φ (SU(2) radial dilation), 0° to 90°:**

![Möbius phi](results/transforms_Rotation_phi.png)

**ψ₁ (planar S¹ rotation), 0° to 90°:**

![Rotation psi1](results/transforms_Rotation_psi1.png)

**Composed φ+ψ₁:**

![Mobius combo](results/transforms_Mobius_combo_phi_psi1.png)

**Scale, 0.5× to 3×:**

![Scale](results/transforms_Scale.png)

**Random projective homographies:**

![Projective](results/transforms_Projectif.png)

**SU(2) matrices drawn from the Haar measure:**

![SU2 Haar](results/transforms_SU2_Haar.png)

### 2.4 Training curves

![Training curves](results/training.png)

Per-epoch history is in [`checkpoints/mnist_v10/history.csv`](checkpoints/mnist_v10/history.csv).
It covers Bi-LogPolar only: the other three models were trained in an earlier
session and reloaded from their checkpoints, so their epoch-level logs were not
regenerated. The final accuracies of all four models are in
`results/table1_clean_accuracy.csv` and are reproduced by Phase A of the
notebook.

---

## 3. What the notebook checks

| Phase | Question it answers | Outputs |
|---|---|---|
| **A** | Do the released weights reproduce the reported accuracies? | `table1_clean_accuracy.csv` |
| **B** | Is equivariance a property of the architecture rather than a by-product of training? | `equivariance_random.csv`, `equivariance_trained.csv`, `equivariance_comparison.png` |
| **C** | Do the tested amplitudes remain semantically valid? | `transforms_*.png` |
| **D** | How does accuracy degrade with amplitude, across six families? | `rob_sweep_*.csv`, `rob_summary.csv`, `robustness_curves.png` |
| **E** | What did the training curves look like? | `training.png`, `history.csv` |
| **F** | Cross-model summary | `reviewer_summary.csv` |

Phase E replots a recorded history. Nothing is retrained.

---

## 4. Mapping from the paper to this repository

| Paper | File | Notebook phase |
|---|---|---|
| Table `<n>` (clean accuracy, parameter counts) | `results/table1_clean_accuracy.csv` | A |
| Table `<n>` (robustness summary) | `results/rob_summary.csv` | D |
| Figure `<n>` (robustness curves) | `results/robustness_curves.png` | D |
| Figure `<n>` (architectural equivariance) | `results/equivariance_random.png` | B |
| Figure `<n>` (equivariance, random vs trained) | `results/equivariance_comparison.png` | B |
| Figure `<n>` (transformation examples) | `results/transforms_*.png` | C |
| Figure `<n>` (training curves) | `results/training.png` | E |
| Appendix `<n>` (per-amplitude sweeps) | `results/rob_sweep_{phi,psi1,scale,proj,combo,su2}.csv` | D |

Each `rob_sweep_*.csv` has one row per amplitude and one column per model, so
any curve in the figures can be re-derived directly from the tables.

---

## 5. Repository layout

```
.
├── Reviewer_Repro_SU2xSim2_MNIST_V10.ipynb      pre-executed, evaluation only
├── checkpoints/mnist_v10/
│   ├── manifest.json                params, accuracy, hyper-parameters, SHA-256
│   ├── {Classic,Sim2Only,SU2Stereo,BiLogPolar}.ts.pt          trained weights
│   ├── {Classic,Sim2Only,SU2Stereo,BiLogPolar}_random.ts.pt   at initialisation
│   ├── equivariance_random.json     fallback if random-init weights are absent
│   └── history.csv                  per-epoch training history
├── results/                         figures and tables from the committed run
├── LICENSE
└── README.md
```

### On the weight format

Checkpoints are TorchScript archives rather than raw `state_dict` files. Every
one of them satisfies

```
forward(x : (B, 1, 64, 64)) -> (logits : (B, 10), features : (B, D))
```

so the notebook can run inference and read internal descriptors without
instantiating any architecture. Each trace was validated numerically against
the eager model at three different batch sizes before being written.

Model definitions and the training loop are not included in this artifact.
They will be released with the camera-ready version.

### On the descriptor used for equivariance

The cosine error of Section 2.2 is measured on the tensor entering the
classification head, which is the same quantity for all four architectures.
This matters because the models do not share a common pooling layer:
`SU2Stereo` has no `AdaptiveAvgPool2d` at all, and the first one in
`BiLogPolar` belongs to its pole estimator rather than to the backbone.

### Integrity

`manifest.json` records the SHA-256 of every checkpoint. The notebook
recomputes them at load time and reports any mismatch, so a reviewer can tell
whether the loaded weights are the ones the authors declared.

---

## 6. Retraining from scratch

Training code is not part of this artifact. Section 7 lists the optimiser
settings, epoch budgets and data pipeline needed to reimplement the protocol,
and `checkpoints/mnist_v10/history.csv` gives the per-epoch history of the
released run.

---

## 7. Protocol notes

Choices a reviewer will want stated explicitly. The notebook surfaces all of
them at run time.

**Evaluation transform.** The pipeline is
`Resize(64) → RandomCrop(64, padding=6) → Normalize(0.1307, 0.3081)`, identical
to the training pipeline, so test images undergo a random ±6 px translation.
The reviewer notebook fixes the seed, making the reported figures reproducible
run to run. An `EVAL_TRANSFORM = "deterministic"` mode (reflect padding plus
centre crop) is provided to measure sensitivity to this choice; it shifts
accuracies by a few tenths of a point.

**Excluded classes.** Classes 1 and 9 are removed from the robustness sweeps.
Under large-amplitude rotations and Möbius transformations digit identity is no
longer well defined (a 6 maps onto a 9), so the ground-truth label becomes
meaningless. Section 2.1 accuracies are therefore on eight classes and are not
directly comparable to the ten-class figures of Phase A.

**Checkpoint selection.** During training the checkpoint maximising accuracy on
the MNIST test split is retained, and that accuracy is what is reported. There
is no separate validation split.

**Runs.** Reported numbers come from a single training run per architecture.

**Run-to-run variance.** Because the evaluation transform applies a random ±6 px
translation, accuracies vary by up to ±0.25 points between evaluation passes at
identical weights. Every figure in Section 2 comes from a single pass.

**Spectral pooling.** The spherical pooling layer pins the maximum azimuthal
order explicitly rather than relying on the `torch-harmonics` default, which
changed across releases. The behaviour of the released checkpoints is therefore
independent of the library version. This is baked into the exported archives.

**Epoch budgets.** Epoch counts differ across models (Classic 20, Sim(2) only
20, SU(2) stereographic 30, Bi-LogPolar 25); each was trained until its
accuracy curve flattened.

---

## 8. Environment

Training and evaluation happened in two separate sessions, so both are
recorded.

**Training** (weights released here):

- Python 3.12
- Single GPU, one run per architecture; epoch counts are in Section 7 and the
  per-epoch history is in `checkpoints/mnist_v10/history.csv`

**Evaluation and export** (the committed notebook run):

- Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128
- NVIDIA L4, CUDA 12.8

Reviewers need neither environment: TorchScript archives are forward
compatible, so the checkpoints load with any PyTorch release at or above the
version recorded in `manifest.json`.

---

## 9. License and citation

The released weights are covered by the BSD 3-Clause License; see
[`LICENSE`](LICENSE). Model source code is not part of this artifact and will
be released with the camera-ready version.

MNIST is used under its original terms and is not redistributed here: the
notebook downloads it via `torchvision.datasets.MNIST`.

Author and citation information is withheld during the double-blind review
period and will be added on acceptance.
