# 🌟 Robust Quality-Diversity (QD) Portfolio Ensemble Optimization 🌟

This repository provides a Streamlit app implementation for *Robust Quality-Diversity Portfolio**, a method co-optimizing thousands of sub-portfolios and assembles them to maximize out-of-sample robustness. 
The method combines cutting-edge techniques such as **Generative Meta-Learning**, **Monte Carlo Optimization**, and sparse portfolio construction using the **SparseMax** layer, scaling to thousands of assets.

## Key Highlights

### 🔹 **Maximizing Behavioral Diversity**
- **Multi-Objective Quality-Diversity Optimization**:
  - Behavioral diversity is measured using return correlations across co-optimized sub-portfolios.
  
- **Ensemble μ-Portfolio Returns**:
  - The **blue line** on the cumulative returns chart represents the Ensemble μ-Portfolio returns.
  - The shaded region depicts the distribution of population returns.

### 🔹 **Sparse Portfolio Optimization**
- **Scalability**:
  - Efficiently scales to **thousands of assets**, surpassing traditional optimization methods.
- **SparseMax Layer**:
  - Ensures sparse portfolio construction, allowing for reduced complexity and improved interpretability.
- **Visualization**:
  - A dynamic pie chart displays μ-Portfolio weights after optimization.

### 🔹 **State-of-the-Art Techniques**
- **Generative Meta-Learning**:
  - Invented for non-convex population-based optimization in high-dimensional spaces.
  - Not implemented in this repo for simplicity, please refer to https://github.com/kayuksel/generative-opt
- **Monte Carlo Optimization**:
  - Uses DropBlock for enhanced robustness against noise.
- **Parallelization**:
  - Fully parallelized with PyTorch for large-scale optimization.

## Published in GECCO 2023 🎉
This approach has been peer-reviewed and published in the **Genetic and Evolutionary Computation Conference (GECCO) 2023**.

📄 **Read the paper**: [https://lnkd.in/d2Zz-Gup](https://arxiv.org/abs/2307.07811)
