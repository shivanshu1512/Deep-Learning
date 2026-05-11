# 🖼️ CNNs with Keras

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Explore **Convolutional Neural Networks** with Keras — from basic building blocks (padding, pooling) to Batch Normalisation, the Functional API, and multi-task learning.

---

## 🧠 What You'll Learn
- `padding='same'` vs `padding='valid'`
- **MaxPooling vs AveragePooling**
- **Batch Normalisation** — faster convergence, regularisation
- **Functional API** — ResNet-style skip connections
- **Multi-task learning** — shared CNN backbone for age + gender

---

## 🚀 Run It

```bash
cd 04_CNNs_with_Keras
pip install -r requirements.txt

python demo.py --demo all         # run all experiments on MNIST
python demo.py --demo batchnorm   # BatchNorm vs no BatchNorm
python demo.py --demo padding     # same vs valid padding
python demo.py --demo pooling     # max vs average pooling
python demo.py --demo functional  # ResNet-style skip connection
```

---

## 📁 Structure

```
04_CNNs_with_Keras/
├── src/
│   └── cnn_keras.py   # All model builders + plot helpers
├── demo.py
├── results/
└── requirements.txt
```

---

## 📈 Generated Plots

| Plot | Description |
|------|-------------|
| `batchnorm_yes/no.png` | Training curves with/without BN |
| `padding_same/valid.png` | Accuracy curves |
| `pooling_max/average.png` | Accuracy curves |
| `functional_api.png` | Skip connection training curves |
