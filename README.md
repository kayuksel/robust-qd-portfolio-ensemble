# 🌟 Robust QD Portfolio Ensemble Optimization 🌟

This repository provides a Streamlit app implementation for **Robust Quality-Diversity Portfolio**, a method co-optimizing thousands of sub-portfolios and assembles them to maximize out-of-sample robustness. 
The method combines cutting-edge techniques such as **Generative Meta-Learning**, **Monte Carlo Optimization**, and sparse portfolio construction using the **SparseMax** layer, scaling to thousands of assets.

## Key Highlights

- **Quality-Diversity Optimization**:
  - Behavioral diversity is measured using return correlations across co-optimized sub-portfolios.
- **Scalability**:
  - Efficiently scales to **thousands of assets**, surpassing traditional optimization methods.
- **SparseMax Layer**:
  - Ensures sparse portfolio construction, allowing for reduced complexity and improved interpretability.
- **Generative Meta-Learning**:
  - Invented for non-convex population-based optimization in high-dimensional spaces.
  - Not included here for simplicity, refer to https://github.com/kayuksel/generative-opt
- **Monte Carlo Optimization**:
  - Uses DropBlock for enhanced robustness against noise.
- **Parallelization**:
  - Fully parallelized with PyTorch for large-scale optimization.

## 🚀 Quick Start Guide

Follow the steps below to get the Streamlit app up and running:

```bash
git clone https://github.com/your-repo/robust-qd-portfolio.git
cd robust-qd-portfolio
pip install -r requirements.txt
streamlit run app.py --server.maxUploadSize 1000
```

## Reward Function Overview

The optimization target is based on the **multiplication of the Probabilistic Sharpe Ratio (PSR)** and the **Omega Ratio (OMG)**, which balances portfolio performance and robustness against extreme returns.

### Secondary Objective

- **Maximum Correlation Penalty**: Ensures sub-portfolios within the population are diversified by reducing inter-correlation.
- **Eigen Entropy Regularization (optional)**: Encourages further out-of-sample robustness of the individual sub-portfolios.

### Modifying the Reward Function

You can modify the **primary objective** (`calculate_psr(rets.T) * omg`) to incorporate new performance metrics. However, it is critical to **retain the secondary objective**, which ensures robustness and diversification.

## Published in GECCO 2023 🎉

📄 **Read the paper**: [https://arxiv.org/abs/2307.07811](https://arxiv.org/abs/2307.07811)
