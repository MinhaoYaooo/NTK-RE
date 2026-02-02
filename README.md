# NTK-RE: Equivalence between neural tangent kernel (NTK) and random-effects model

## Project Structure

```text
NTK-RE/
├── environment.yml          # Conda environment configuration
├── train.py                 # Main training & comparison script
├── src/                     # Core library code
│   ├── model.py             # Dynamic Depth/Width DNN
│   ├── ntk.py               # Jacobian contraction & NTK computation
│   ├── reml.py              # REML optimization logic
│   └── hypothesis_test.py   # Statistical significance testing
└── README.md
```

---
## Installation

1.  **Clone or download** this repository.
    ```bash
    git clone https://github.com/MinhaoYaooo/NTK-RE
    cd NTK-RE
    ```
3.  **Create the Conda environment**:
    ```bash
    conda env create -f environment.yml
    ```
4.  **Activate the environment**:
    ```bash
    conda activate NTK-RE
    ```

---
## Quick Start: Data Generation

Before running the examples, let's generate a dataset using the **"Simple Function"** (Case 1) defined in our simulations.

Create a file named `generate_data.py` in the root folder and run it:

```python
import torch
import numpy as np

# Define the "Simple" Ground Truth Function
def f_true_simple(X):
    y = 0.5 * X[:, 0:1]
    y += 0.8 * torch.tanh(X[:, 1:2])
    y += torch.sin(X[:, 2:3]) 
    y += 0.6 * X[:, 3:4]
    y += 0.3 * X[:, 4:5]**2
    y += 0.05 * torch.exp(X[:, 5:6]) 
    y += torch.cos(X[:, 6:7])
    y += 0.5 * torch.abs(X[:, 7:8])
    y += 0.4 * X[:, 8:9]
    y += 0.7 * torch.sin(X[:, 9:10])
    return 0.2 * y

# 1. Setup
torch.manual_seed(42)
n_samples = 600
p = 10

# 2. Generate Data
X = torch.randn(n_samples, p)
Y = f_true_simple(X) + torch.randn(n_samples, 1) * 0.5  # Add noise

# 3. Save Tensors
torch.save(X, 'data_X.pt')
torch.save(Y, 'data_Y.pt')

print("Data saved to data_X.pt and data_Y.pt")
```

---

## Usage Method 1: Command Line Interface (CLI)

You can run the full training pipeline directly from your terminal. This will load your data, compute the REML stopping time, run GD and NTK Flow, and save the results.

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

**Outputs (in `./experiments/simple_case`):**
* `comparison_plot.png`: Visual comparison of Train/Test errors.
* `training_report.txt`: Summary of Best $t$, REML $t$, and Hypothesis Test p-values.
* `errors_gd.csv` / `errors_ntk.csv`: Raw training trajectory data.

---

## Usage Method 2: Python API

You can also import the `train` function into your own scripts for more flexible experimentation.

```python
import torch
from train import train

# 1. Load your tensors
X = torch.load('data_X.pt')
Y = torch.load('data_Y.pt')

# 2. Run the training pipeline
# This will perform Hypothesis Testing, REML calc, and Training automatically.
train(
    X=X, 
    Y=Y, 
    width=500,        # Network width
    depth=2,          # Network depth
    max_t=50000,      # Max epochs
    lr=1e-2,          # Learning rate
    output_dir='./experiments/python_api_run',
    split_ratio=0.8   # Train/Test split
)
```




