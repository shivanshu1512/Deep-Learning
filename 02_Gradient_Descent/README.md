# 📉 Gradient Descent — From Scratch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Compare **Batch GD vs SGD vs Mini-Batch GD** from scratch. Visualise the **vanishing gradient** problem and see how learning rate affects convergence.

---

## 🧠 What You'll Learn
- **Batch Gradient Descent** — stable but slow
- **Stochastic GD** — fast but noisy  
- **Mini-Batch GD** — best of both worlds
- **Vanishing Gradient** through sigmoid activations
- Effect of learning rate on convergence

---

## 🚀 Run It

```bash
cd 02_Gradient_Descent
pip install -r requirements.txt

python demo.py                     # all demos
python demo.py --demo compare      # GD variant comparison
python demo.py --demo vanishing    # vanishing gradient plot
python demo.py --demo lr           # learning rate effect
```

---

## 📁 Structure

```
02_Gradient_Descent/
├── src/
│   └── gradient_descent.py  # All GD variants + vanishing gradient
├── demo.py
├── results/
└── requirements.txt
```

---

## 📈 Generated Plots

| Plot | Description |
|------|-------------|
| `gd_comparison.png` | Batch vs SGD vs Mini-Batch loss curves |
| `vanishing_gradient.png` | Gradient magnitude decay per layer |
| `lr_effect.png` | Convergence at different learning rates |
