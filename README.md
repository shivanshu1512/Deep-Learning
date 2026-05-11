# 🧠 Deep Learning — From Scratch to Keras

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Only%20(scratch)-013243?style=for-the-badge&logo=numpy)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.12%2B-D00000?style=for-the-badge&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A structured, from-scratch Deep Learning curriculum — building every concept by hand before using frameworks.**

</div>

---

## 📚 Learning Path

```
01 → Perceptron (the building block)
02 → Gradient Descent (how networks learn)
03 → Backpropagation (how gradients flow)
04 → CNNs with Keras (image understanding)
05 → RNNs & Sentiment (sequence modelling)
```

---

## 📂 Projects

| # | Topic | Key Concepts | Tools |
|---|-------|-------------|-------|
| [01](./01_Perceptron/) | **Perceptron** | Learning rule, hinge loss, XOR limitation | NumPy only |
| [02](./02_Gradient_Descent/) | **Gradient Descent** | Batch vs SGD vs Mini-batch, vanishing gradient, LR effect | NumPy only |
| [03](./03_Backpropagation/) | **Backpropagation** | Chain rule, He/Xavier init, momentum SGD, classification + regression | NumPy only |
| [04](./04_CNNs_with_Keras/) | **CNNs with Keras** | Padding, pooling, BatchNorm, Functional API, multi-task (age+gender) | TensorFlow/Keras |
| [05](./05_RNNs_and_Sentiment/) | **RNNs & Sentiment** | SimpleRNN, BiLSTM, Deep RNN, IMDB sentiment | TensorFlow/Keras |

---

## 🏗️ Repository Structure

```
Deep-Learning/
│
├── 01_Perceptron/
│   ├── src/perceptron.py    # Perceptron class + hinge loss + plots
│   ├── demo.py              # python demo.py [--dataset linear|xor|all]
│   └── requirements.txt
│
├── 02_Gradient_Descent/
│   ├── src/gradient_descent.py  # Batch/SGD/Mini-batch + vanishing GD
│   ├── demo.py                  # python demo.py [--demo compare|vanishing|lr]
│   └── requirements.txt
│
├── 03_Backpropagation/
│   ├── src/backprop.py     # Full MLP: forward + backward + momentum SGD
│   ├── demo.py             # python demo.py [--task classification|regression]
│   └── requirements.txt
│
├── 04_CNNs_with_Keras/
│   ├── src/cnn_keras.py    # Padding/Pooling/BN/Functional API/Age-Gender CNN
│   ├── demo.py             # python demo.py [--demo batchnorm|padding|pooling|functional]
│   └── requirements.txt
│
├── 05_RNNs_and_Sentiment/
│   ├── src/rnn_sentiment.py  # SimpleRNN, BiLSTM, DeepRNN on IMDB
│   ├── demo.py               # python demo.py [--model bilstm] [--compare]
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/shivanshu1512/Deep-Learning.git
cd Deep-Learning

# Projects 01-03: NumPy only
pip install numpy matplotlib
cd 01_Perceptron && python demo.py

# Projects 04-05: TensorFlow required
pip install tensorflow matplotlib
cd 04_CNNs_with_Keras && python demo.py
```

---

## 📖 What Each Project Demonstrates

### 01 — Perceptron (NumPy from scratch)
- Complete Perceptron learning rule implementation
- **Hinge loss** function and visualisation
- Why Perceptrons **cannot solve XOR** — the fundamental limitation that led to multi-layer networks
- Decision boundary animation

### 02 — Gradient Descent (NumPy from scratch)
- **Batch GD vs SGD vs Mini-Batch GD** — convergence comparison
- **Vanishing gradient** through sigmoid activations (visual proof)
- **Learning rate effect** — too small (slow), too large (diverges)

### 03 — Backpropagation (NumPy from scratch)
- Full MLP with analytical backprop — **no autograd**
- **He initialisation** (ReLU) and **Xavier initialisation** (Sigmoid)
- **Momentum SGD** to escape local minima
- Decision boundary for classification + curve fitting for regression

### 04 — CNNs with Keras
- `padding='same'` vs `padding='valid'` — output size comparison
- **MaxPooling vs AveragePooling** on MNIST
- **Batch Normalisation**: with vs without — convergence speed
- **Functional API**: ResNet-style skip connection
- **Multi-task CNN**: shared backbone → age regression + gender classification

### 05 — RNNs & Sentiment Analysis
- **SimpleRNN** baseline on IMDB
- **Bidirectional LSTM** (best accuracy)
- **Deep stacked RNN** (3 layers)
- EarlyStopping + ReduceLROnPlateau callbacks
- Model comparison table

---

## 🛠️ Technologies

| Projects | Libraries |
|----------|-----------|
| 01, 02, 03 | `numpy`, `matplotlib` (from scratch) |
| 04, 05 | `tensorflow`, `keras`, `matplotlib` |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/shivanshu1512">shivanshu1512</a> while learning Deep Learning
</div>
