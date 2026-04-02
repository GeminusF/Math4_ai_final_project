# Math4AI Final Capstone — Team GradienMindt

**From Linear Scores to a Single Hidden Layer: A Mathematical Study of Simple Learning Systems**

National AI Center · AI Academy

---

## Project Overview

This capstone investigates a single, precise question:

> *When does a one-hidden-layer nonlinear classifier genuinely improve on a linear decision rule, and when is additional model complexity unnecessary?*

We implement two classifiers **from scratch in NumPy**:

| Model | Description |
|---|---|
| **Softmax Regression** | Multiclass linear classifier; affine scores passed through softmax |
| **One-Hidden-Layer Neural Network** | `tanh` hidden activations, softmax output, trained via backpropagation |

We compare them on three datasets (two synthetic, one benchmark) and evaluate whether the added complexity of the neural network is justified by the evidence.

---

## Team

| Name | Role |
|---|---|
| **Milana Karimova** | Training loop and best-validation checkpointing (`train.py`); SGD, Momentum, and Adam (`optimizers.py`); evaluation and repeated-seed CIs (`evaluate.py`); 5-seed runs and Track B confidence/reliability (`run_repeated_seeds.py`, `run_advanced_track.py`); final slide deck |
| **Fereh Feyzullayev** | Softmax regression (`softmax_regression.py`); plotting and reliability visuals (`visualize.py`); synthetic experiments (`run_synthetic.py`); LaTeX report and slide compilation |
| **Nijat Aghayev** | Stable softmax, cross-entropy, and helpers (`math_utils.py`); data loading (`data_loader.py`); one-hidden-layer network and backprop (`neural_network.py`); digits and ablation runs (`run_digits.py`, `run_ablations.py`); results summary; co-authored report |

---

## Repository Structure

```
math4ai_final_capstone_gradienmindt/
├── README.md                          ← this file
├── deliverables/
│   └── math4ai_capstone_assignment.tex  ← official assignment handout
└── starter_pack/
    ├── README.md                      ← starter-pack overview
    ├── CHECKLIST.md                   ← setup and submission checklist
    ├── data/
    │   ├── digits_data.npz            ← digits features (64-dim) and labels
    │   ├── digits_split_indices.npz   ← fixed train/val/test indices
    │   ├── linear_gaussian.npz        ← linear synthetic dataset
    │   └── moons.npz                  ← nonlinear synthetic dataset
    ├── scripts/
    │   ├── make_digits_split.py       ← deterministic split generator
    │   └── generate_synthetic.py      ← regenerates linear_gaussian and moons
    ├── src/
    │   ├── data_loader.py             ← data loading utilities
    │   ├── softmax_regression.py      ← softmax regression model
    │   ├── neural_network.py          ← one-hidden-layer network with tanh
    │   ├── optimizers.py              ← SGD, Momentum, Adam
    │   ├── math_utils.py              ← stable softmax, cross-entropy helpers
    │   ├── train.py                   ← mini-batch training loop
    │   ├── evaluate.py                ← accuracy and cross-entropy metrics
    │   └── visualize.py               ← decision-boundary and training plots
    ├── experiments/
    │   ├── run_digits.py              ← core digits comparison
    │   ├── run_synthetic.py           ← synthetic task comparisons
    │   ├── run_ablations.py           ← capacity ablation on moons
    │   ├── run_repeated_seeds.py      ← 5-seed evaluation on digits
    │   └── run_advanced_track.py      ← advanced track analysis
    ├── figures/                       ← saved plots
    ├── results/                       ← saved metrics and summary tables
    ├── report/
    │   └── report.pdf                 ← final submitted report
    └── slides/
        └── slide.pdf                  ← presentation slides
```

---

## Environment Setup

**Requirements:** Python 3.8+, NumPy, Matplotlib. No deep-learning frameworks are used.

```bash
# Clone the repository
git clone https://github.com/Murad-Huseynli/math4ai_capstone.git
cd math4ai_capstone

# Install dependencies
pip install numpy matplotlib
```

No additional setup is needed. All data files are included in `starter_pack/data/`.

---

## Reproducing the Experiments

All experiment scripts are in `starter_pack/experiments/`. Run them from the repository root.

### 1. Synthetic Tasks (linear Gaussian + moons)

```bash
python starter_pack/experiments/run_synthetic.py
```

Produces decision-boundary plots for both models on both datasets. Figures are saved to `starter_pack/figures/`.

### 2. Core Digits Benchmark

```bash
python starter_pack/experiments/run_digits.py
```

Trains and evaluates softmax regression and the neural network on the fixed digits split. Reports accuracy and mean cross-entropy on train, validation, and test sets.

### 3. Capacity Ablation (moons, hidden widths 2 / 8 / 32)

```bash
python starter_pack/experiments/run_ablations.py
```

### 4. Optimizer Study (SGD / Momentum / Adam on digits, hidden width 32)

Included in `run_digits.py` via a flag, or run standalone:

```bash
python starter_pack/experiments/run_digits.py --optimizer_study
```

### 5. Repeated-Seed Evaluation (5 seeds on digits)

```bash
python starter_pack/experiments/run_repeated_seeds.py
```

Reports mean ± 95% CI for test accuracy and test cross-entropy over 5 seeds for both models.

### 6. Advanced Track

```bash
python starter_pack/experiments/run_advanced_track.py
```

### Regenerating the Synthetic Data (optional)

The provided `.npz` files are the fixed inputs used in all experiments. To verify reproducibility:

```bash
python starter_pack/scripts/generate_synthetic.py
python starter_pack/scripts/make_digits_split.py
```

---

## Fixed Experimental Protocol

To ensure fair comparison, all experiments use the same fixed protocol:

| Setting | Value |
|---|---|
| Train/val/test split | Fixed indices in `digits_split_indices.npz` |
| Digits input | 64-dim flattened vectors, pre-scaled to \[0, 1\] |
| Hidden width (default) | 32 |
| L2 regularization | λ = 1e-4 (both models) |
| Mini-batch size | 64 |
| Epoch budget | 200 (best val cross-entropy checkpoint) |
| SGD learning rate | 0.05 |
| Momentum lr / β | 0.05 / 0.9 |
| Adam lr / β₁ / β₂ / ε | 0.001 / 0.9 / 0.999 / 1e-8 |
| Repeated seeds | 5 seeds, 95% CI = x̄ ± 2.776 · s/√5 |

Model selection is performed using **validation cross-entropy only**. The test set is evaluated exactly once per seed after configuration selection.

---

## Deliverables

| Deliverable | Location |
|---|---|
| Report (PDF) | `starter_pack/report/report.pdf` |
| Slides (PDF) | `starter_pack/slides/slide.pdf` |
| Assignment handout | `deliverables/math4ai_capstone_assignment.tex` |

---

## What Is and Is Not Included

The starter pack provides fixed data and deterministic scripts only. All model code, training loops, optimizer implementations, backpropagation, and plotting code are written by the team.

**Not used:** PyTorch, TensorFlow, JAX, autograd, scikit-learn model classes.
