#!/usr/bin/env python3
import argparse
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- IMPORTS ---
from src.model import DNN
from src.ntk import get_ntk_matrices
from src.reml import compute_reml_t
# NEW IMPORT
from src.hypothesis_test import test_initialization 

def train(X, Y, width=500, depth=1, max_t=100000, lr=1e-2, output_dir='./results', split_ratio=0.8):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Data Prep ---
    n_total = X.shape[0]
    n_train = int(n_total * split_ratio)
    
    indices = torch.randperm(n_total)
    X_train, y_train = X[indices[:n_train]], Y[indices[:n_train]]
    X_test, y_test = X[indices[n_train:]], Y[indices[n_train:]]
    
    print(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    
    # --- 2. Initialize Model ---
    p = X_train.shape[1]
    model = DNN(input_dim=p, width=width, depth=depth)
    
    with torch.no_grad():
        f0_train = model(X_train)
        f0_test = model(X_test)
        
    # --- 3. NTK Computation ---
    print("Computing NTK matrices...")
    H, C, fnet, params = get_ntk_matrices(model, X_train, X_test)
    
    # --- 4. Hypothesis Testing (NEW) ---
    print("Performing Hypothesis Test (Need for Training)...")
    pval, pval_asymp = test_initialization(H, y_train, f0_train)
    print(f"P-value: {pval:.4e} (Asymp: {pval_asymp:.4e})")
    
    # --- 5. REML Calculation ---
    print("Calculating REML stopping time...")
    t_reml = compute_reml_t(H, y_train, f0_train, lr)
    print(f"REML calculated t: {t_reml:.4f}")

    # --- 6. Gradient Descent (GD) ---
    print(f"Running Gradient Descent ({max_t} epochs)...")
    optimizer = optim.SGD(model.parameters(), lr=lr)
    gd_history = [] 

    for step in range(max_t):
        epoch = step + 1
        optimizer.zero_grad()
        loss = 0.5 * torch.mean((model(X_train) - y_train) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss = 0.5 * torch.mean((model(X_train) - y_train) ** 2).item()
            test_loss = 0.5 * torch.mean((model(X_test) - y_test) ** 2).item()
        
        gd_history.append({'t': epoch, 'train_error': train_loss, 'test_error': test_loss})
        
        if epoch % (max_t // 10) == 0:
            print(f"GD Epoch {epoch}: Train Loss {train_loss:.4f}")

    # --- 7. NTK Gradient Flow ---
    print("Computing NTK Gradient Flow...")
    ntk_history = []
    epoch_ntk_steps = np.unique(np.round(np.logspace(np.log10(1), np.log10(max_t), 100)).astype(int))
    
    H_inv = torch.linalg.inv(H)
    eye_n = torch.eye(n_train)
    
    for t_val in epoch_ntk_steps:
        exp_term = torch.matrix_exp(-t_val * lr * H)
        diff = (eye_n - exp_term) @ (y_train - f0_train)
        
        y_ntk_train_pred = f0_train + diff
        y_ntk_test_pred = f0_test + C @ H_inv @ diff
        
        tr_err = 0.5 * torch.mean((y_ntk_train_pred - y_train) ** 2).item()
        te_err = 0.5 * torch.mean((y_ntk_test_pred - y_test) ** 2).item()
        ntk_history.append({'t': t_val, 'train_error': tr_err, 'test_error': te_err})

    # --- 8. Save CSVs ---
    df_gd = pd.DataFrame(gd_history)
    df_ntk = pd.DataFrame(ntk_history)
    df_gd.to_csv(os.path.join(output_dir, 'errors_gd.csv'), index=False)
    df_ntk.to_csv(os.path.join(output_dir, 'errors_ntk.csv'), index=False)

    # --- 9. Metrics & Reporting ---
    best_row_gd = df_gd.loc[df_gd['test_error'].idxmin()]
    best_row_ntk = df_ntk.loc[df_ntk['test_error'].idxmin()]
    
    idx_reml_gd = (np.abs(df_gd['t'] - t_reml)).argmin()
    reml_row_gd = df_gd.iloc[idx_reml_gd]
    
    idx_reml_ntk = (np.abs(df_ntk['t'] - t_reml)).argmin()
    reml_row_ntk = df_ntk.iloc[idx_reml_ntk]
    
    report = [
        "--- TRAINING REPORT ---",
        f"Hypothesis Test P-value: {pval:.6f}",
        f"Hypothesis Test P-value (Asymptotic): {pval_asymp:.6f}",
        f"Decision (alpha=0.05): {'TRAINING REQUIRED' if pval < 0.05 else 'NO SIGNAL DETECTED'}",
        "",
        f"REML Calculated Optimal Stop (t): {t_reml:.4f}",
        "",
        "--- Gradient Descent Results ---",
        f"Actual Best Test Error: {best_row_gd['test_error']:.6f} at Epoch {int(best_row_gd['t'])}",
        f"Error at REML t (~{int(reml_row_gd['t'])}): {reml_row_gd['test_error']:.6f}",
        "",
        "--- NTK Gradient Flow Results ---",
        f"Actual Best Test Error: {best_row_ntk['test_error']:.6f} at Epoch {int(best_row_ntk['t'])}",
        f"Error at REML t (~{int(reml_row_ntk['t'])}): {reml_row_ntk['test_error']:.6f}",
        ""
    ]
    
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print("\n".join(report))

    # --- 10. Plotting ---
    plt.figure(figsize=(12, 5))
    
    def plot_curve(ax, title, df_g, df_n, col):
        ax.plot(df_g['t'], df_g[col], label='Gradient Descent', color='blue' if 'train' in col else 'red')
        ax.plot(df_n['t'], df_n[col], label='NTK Flow', linestyle='--', color='blue' if 'train' in col else 'red')
        ax.set_title(title)
        ax.set_xlabel('Epoch (log)')
        ax.set_ylabel('MSE')
        ax.set_xscale('log')
        ax.axvline(x=t_reml, color='green', linestyle=':', linewidth=2, label=f't_REML ({t_reml:.2f})')
        ax.legend()
        ax.grid(True)

    plt.subplot(1, 2, 1)
    plot_curve(plt.gca(), 'Training Error', df_gd, df_ntk, 'train_error')
    
    plt.subplot(1, 2, 2)
    plot_curve(plt.gca(), 'Test Error', df_gd, df_ntk, 'test_error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_plot.png'), dpi=300)
    print(f"Results saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="NTK-RE Training")
    parser.add_argument('--X', type=str, required=True, help="Path to X tensor")
    parser.add_argument('--Y', type=str, required=True, help="Path to Y tensor")
    parser.add_argument('--width', type=int, default=500)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--max_t', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--split_ratio', type=float, default=0.8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        X_tensor = torch.load(args.X)
        Y_tensor = torch.load(args.Y)
    except FileNotFoundError:
        print(f"Error: Could not find file {args.X} or {args.Y}")
        exit(1)

    train(X=X_tensor, Y=Y_tensor, width=args.width, depth=args.depth, 
          max_t=args.max_t, lr=args.lr, output_dir=args.output_dir, 
          split_ratio=args.split_ratio)