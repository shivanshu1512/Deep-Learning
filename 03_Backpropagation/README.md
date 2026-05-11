# ⚡ Backpropagation — From Scratch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A fully hand-coded **Multi-Layer Perceptron** with manual backpropagation — no autograd, no frameworks. Understand exactly how gradients flow through a network.

---

## 🧠 What You'll Learn
- Forward pass through arbitrary-depth networks
- **Chain rule** applied manually (backprop)
- **He initialisation** (ReLU) vs **Xavier initialisation** (Sigmoid)
- **Momentum SGD** to accelerate convergence
- **ReLU vs Sigmoid** activation functions
- Binary Cross-Entropy (classification) and MSE (regression)

---

## 🚀 Run It

```bash
cd 03_Backpropagation
pip install -r requirements.txt

python demo.py                          # both tasks
python demo.py --task classification    # 2-D Gaussian clusters
python demo.py --task regression        # sin curve fitting
python demo.py --activation sigmoid     # use sigmoid instead of ReLU
```

---

## 📁 Structure

```
03_Backpropagation/
├── src/
│   └── backprop.py   # MLP with full forward + backward pass
├── demo.py
├── results/
└── requirements.txt
```

---

## 📈 Generated Plots

| Plot | Description |
|------|-------------|
| `classification_loss.png` | BCE loss per epoch |
| `classification_boundary.png` | Learned decision boundary |
| `regression_loss.png` | MSE loss per epoch |
| `regression_fit.png` | MLP fit vs sin(x) |
