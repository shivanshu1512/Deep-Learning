# 🔵 Perceptron — From Scratch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Build a **Perceptron from scratch** using only NumPy. Understand the learning rule, hinge loss, and — crucially — why the Perceptron fails on the XOR problem (which led to multi-layer networks).

---

## 🧠 What You'll Learn
- Perceptron learning algorithm (Rosenblatt, 1958)
- The **Perceptron Trick** (weight update rule)
- **Hinge loss** function
- Why Perceptrons **cannot solve non-linearly separable data** (XOR)

---

## 🚀 Run It

```bash
cd 01_Perceptron
pip install -r requirements.txt

python demo.py                    # all demos
python demo.py --dataset linear   # converges perfectly
python demo.py --dataset xor      # shows failure mode
```

---

## 📁 Structure

```
01_Perceptron/
├── src/
│   └── perceptron.py   # Perceptron class + hinge loss + plot functions
├── demo.py             # Entry point
├── results/            # Generated plots
└── requirements.txt
```

---

## 📈 Generated Plots

| Plot | Description |
|------|-------------|
| `linear_boundary.png` | Decision boundary (converges) |
| `xor_boundary.png` | XOR failure (can't separate) |
| `hinge_loss.png` | Hinge loss curve |
| `*_errors.png` | Misclassification count per epoch |
