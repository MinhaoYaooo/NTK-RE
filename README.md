# GF-REML: A Python Toolkit for Random-Effects Inference and Early Stopping of Fixed-Operator Gradient Flows

This repository provides **GF-REML**, a Python toolkit for studying **Fixed-Operator Gradient Flows** and **REML-guided early stopping** in deep neural networks. The toolkit then uses **restricted maximum likelihood (REML)** to estimate principled stopping times and support statistical inference under a random-effects interpretation.

The main goal is to provide a pipeline for:

- Building fully connected DNNs with configurable depth and width
- Computing empirical NTKs through Jacobian contraction
- Comparing finite-width gradient descent with NTK-flow dynamics
- Estimating REML-based early stopping times
- Performing hypothesis tests for learned random effects
- Saving training curves, inference summaries, and comparison plots

---

## Project Structure

```text
GF-REML/
├── environment.yml          # Conda environment configuration
├── LICENSE                  # GPL-3.0 copyleft license
├── train.py                 # Main training, NTK, REML, and comparison pipeline
├── src/                     # Core library code
│   ├── model.py             # Dynamic depth/width neural network model
│   ├── ntk.py               # Jacobian computation and NTK construction
│   ├── reml.py              # REML optimization and stopping-time estimation
│   └── hypothesis_test.py   # Statistical hypothesis testing utilities
└── README.md                # Project documentation
```

---

## Main Components

### Neural Network Model

The neural network architecture is implemented in `src/model.py`.

It supports configurable:

- Input dimension
- Hidden layer width
- Network depth
- Output dimension
- Activation structure

This makes it easy to study how NTK behavior changes as the network becomes wider or deeper.

### Neural Tangent Kernel Computation

The NTK-related utilities are implemented in `src/ntk.py`.

The code computes the empirical NTK by:

1. Evaluating the neural network on the input data
2. Computing Jacobians with respect to model parameters
3. Contracting Jacobians to form the kernel matrix

The NTK matrix is then used to simulate the corresponding kernel-flow predictor.

### REML Estimation

The REML logic is implemented in `src/reml.py`.

REML is used to connect the NTK formulation with a random-effects model interpretation. In this project, it is used to estimate quantities related to regularization, variance components, and stopping behavior.

### Hypothesis Testing

Statistical testing utilities are implemented in `src/hypothesis_test.py`.

These functions are used to evaluate whether learned effects are statistically significant under the random-effects framework.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MinhaoYaooo/GF-REML
cd GF-REML
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
```

### 3. Activate the Environment

```bash
conda activate GF-REML
```

### 4. Verify Installation

You can check that the main dependencies are available by running:

```bash
python -c "import torch; import numpy; print('Environment ready')"
```

---

## Quick Start: Generate Synthetic Data

Before running the training pipeline, generate a simple synthetic regression dataset.

Create a file named `generate_data.py` in the project root:

```python
import torch

def f_true_simple(X):
    y = 0.5 * X[:, 0:1]
    y += 0.8 * torch.tanh(X[:, 1:2])
    y += torch.sin(X[:, 2:3])
    y += 0.6 * X[:, 3:4]
    y += 0.3 * X[:, 4:5] ** 2
    y += 0.05 * torch.exp(X[:, 5:6])
    y += torch.cos(X[:, 6:7])
    y += 0.5 * torch.abs(X[:, 7:8])
    y += 0.4 * X[:, 8:9]
    y += 0.7 * torch.sin(X[:, 9:10])
    return 0.2 * y

torch.manual_seed(42)

n_samples = 600
p = 10

X = torch.randn(n_samples, p)
Y = f_true_simple(X) + torch.randn(n_samples, 1) * 0.5

torch.save(X, "data_X.pt")
torch.save(Y, "data_Y.pt")

print("Data saved to data_X.pt and data_Y.pt")
```

Run:

```bash
python generate_data.py
```

This creates:

```text
data_X.pt
data_Y.pt
```

---

## Usage Method 1: Command Line Interface

Run the full experiment pipeline from the terminal:

```bash
python train.py \
  --X data_X.pt \
  --Y data_Y.pt \
  --width 500 \
  --depth 2 \
  --max_t 50000 \
  --lr 0.01 \
  --output_dir ./experiments/simple_case
```

This command will:

1. Load the input and response tensors
2. Split the data into training and test sets
3. Initialize the neural network
4. Compute the empirical NTK
5. Estimate the REML-based stopping time
6. Train the finite-width neural network using gradient descent
7. Run the corresponding NTK-flow comparison
8. Perform hypothesis testing
9. Save plots, reports, and raw error curves

---

## Command Line Arguments

| Argument | Description | Example |
|---|---|---|
| `--X` | Path to input tensor file | `data_X.pt` |
| `--Y` | Path to response tensor file | `data_Y.pt` |
| `--width` | Hidden layer width of the neural network | `500` |
| `--depth` | Number of hidden layers | `2` |
| `--max_t` | Maximum training iterations or time horizon | `50000` |
| `--lr` | Learning rate for gradient descent | `0.01` |
| `--output_dir` | Directory where results are saved | `./experiments/simple_case` |

---

## Output Files

After running the CLI command, the output directory will contain files such as:

```text
experiments/simple_case/
├── comparison_plot.png
├── training_report.txt
├── errors_gd.csv
└── errors_ntk.csv
```

### `comparison_plot.png`

A visual comparison of training and test error curves for:

- Finite-width neural network gradient descent
- NTK-flow approximation
- REML-selected stopping time

### `training_report.txt`

A text summary containing key experimental results, such as:

- Best empirical stopping time
- REML-estimated stopping time
- Train and test errors
- Hypothesis testing results
- p-values and statistical summaries

### `errors_gd.csv`

Raw error trajectory for the finite-width neural network trained by gradient descent.

### `errors_ntk.csv`

Raw error trajectory for the NTK-flow approximation.

---

## Usage Method 2: Python API

You can also call the training pipeline directly from Python.

```python
import torch
from train import train

X = torch.load("data_X.pt")
Y = torch.load("data_Y.pt")

train(
    X=X,
    Y=Y,
    width=500,
    depth=2,
    max_t=50000,
    lr=1e-2,
    output_dir="./experiments/python_api_run",
    split_ratio=0.8,
)
```

This is useful when you want to run multiple experiments programmatically, tune hyperparameters, or integrate the pipeline into a larger research workflow.

---

## Example Experiment Workflow

A typical experiment may look like this:

```bash
python generate_data.py

python train.py \
  --X data_X.pt \
  --Y data_Y.pt \
  --width 500 \
  --depth 2 \
  --max_t 50000 \
  --lr 0.01 \
  --output_dir ./experiments/simple_width500_depth2
```

You can repeat the experiment with different network widths:

```bash
python train.py --X data_X.pt --Y data_Y.pt --width 100  --depth 2 --max_t 50000 --lr 0.01 --output_dir ./experiments/width100
python train.py --X data_X.pt --Y data_Y.pt --width 500  --depth 2 --max_t 50000 --lr 0.01 --output_dir ./experiments/width500
python train.py --X data_X.pt --Y data_Y.pt --width 1000 --depth 2 --max_t 50000 --lr 0.01 --output_dir ./experiments/width1000
```

This can help investigate whether wider neural networks behave more similarly to their NTK approximation.

---

## Data Format

The training script expects PyTorch tensor files.

### Input Tensor

```python
X.shape == (n_samples, n_features)
```

Example:

```python
X.shape == (600, 10)
```

### Response Tensor

```python
Y.shape == (n_samples, 1)
```

Example:

```python
Y.shape == (600, 1)
```

Both tensors should be saved using:

```python
torch.save(X, "data_X.pt")
torch.save(Y, "data_Y.pt")
```

---


## Troubleshooting

### Conda environment creation fails

Make sure Conda is installed and up to date:

```bash
conda --version
conda update conda
```

Then retry:

```bash
conda env create -f environment.yml
```

### CUDA or GPU issues

If PyTorch cannot find CUDA, confirm your installed PyTorch version matches your CUDA version. The code can also be run on CPU, although NTK computation may be slower for large datasets or wide networks.

### Out-of-memory errors

Computing the NTK can be memory intensive because it involves Jacobians and kernel matrices. If you encounter memory issues, try reducing:

- Number of samples
- Network width
- Network depth
- Maximum training horizon

### Slow NTK computation

NTK computation may be expensive for large models. Start with a small experiment first, such as:

```bash
python train.py \
  --X data_X.pt \
  --Y data_Y.pt \
  --width 100 \
  --depth 2 \
  --max_t 5000 \
  --lr 0.01 \
  --output_dir ./experiments/debug_run
```

---

## Suggested Citation

If you use this repository in academic work, please cite it as:

```bibtex
@misc{yao2026gradientflowoptimizationdynamicrandomeffects,
      title={Gradient-Flow Optimization as Dynamic Random-Effects Inference: Testing and Early Stopping with Applications to Deep Learning}, 
      author={Minhao Yao and Ruoyu Wang and Xihong Lin and Lin Liu and Zhonghua Liu},
      year={2026},
      eprint={2605.27991},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.27991}, 
}
```

---

## Authorship and copyright

**Python implementation:** Minhao Yao
**Copyright:** Copyright (C) 2026 Minhao Yao.
**License:** GPL-3.0.

## License

This package is distributed under the GNU General Public License v3.0. See the LICENSE file for details.

---
