# Evolution Strategies as a Scalable Alternative to Reinforcement Learning

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.4.0-orange.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

A modular, clean, and high-performance Python implementation of **OpenAI Evolution Strategies (ES)** based on the seminal paper by *Salimans et al. (2017)*. Developed as a course project for **Evolutionary Computation**, this repository benchmarks ES against **Random Hill Climbing (RHC / [1+1]-ES)** and **REINFORCE** across discrete (`CartPole-v1`) and continuous (`Pendulum-v1`) control environments in OpenAI Gymnasium.

---

## 🚀 Key Features & Algorithmic Alignment

1. **Parameter-Space Smoothing**: Unlike standard policy gradients that add noise in action space, ES smooths the objective function by perturbing policy parameters under an isotropic multivariate Gaussian distribution $\mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} F(\theta + \sigma \epsilon)$[cite: 2, 3, 5].
2. **Shared Noise Table & Scalar Communication (Algorithm 2)**: To eliminate inter-process communication bottlenecks, parallel workers initialize an identical pseudo-random number generator seed. Workers evaluate perturbations locally and communicate strictly scalar episodic returns[cite: 1, 3].
3. **Antithetic (Mirrored) Sampling**: Evaluates mirrored pairs $(\theta + \sigma\epsilon, \theta - \sigma\epsilon)$ on matching environment seeds to drastically lower gradient estimation variance[cite: 4].
4. **Rank Fitness Shaping & Weight Decay**: Transforms raw returns into centered ranks in $[-0.5, 0.5]$ to prevent premature convergence from outlier trajectories, accompanied by L2 weight decay regularization[cite: 4].
5. **Dynamic Episode Capping**: Dynamically adjusts maximum episode length $m = 2 \times \text{mean\_steps}$ to maintain high CPU utilization throughout training[cite: 4].
6. **Virtual Batch Normalization & Observation Normalization**: Implements state observation scaling and dynamic Virtual Batch Normalization to ensure early-stage parameter perturbations yield diverse exploratory behaviors[cite: 1, 4].

---

## 📂 Project Structure

```text
├── models.py         # PyTorch policy networks, observation normalization & VBN
├── algorithms.py     # Parallel OpenAI-ES (Algorithm 2), RHC, and REINFORCE
├── visualization.py  # Learning curves, 95% CIs, Mann-Whitney U tests & MP4 video recording
├── main.py           # CLI entry point for benchmarks, ablations, scaling & frame-skip studies
├── requirements.txt  # Project dependency specifications
└── outputs/          # Generated plots, numerical summaries, and agent MP4 videos