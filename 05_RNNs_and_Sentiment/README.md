# 📝 RNNs & Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

Sentiment analysis on **IMDB movie reviews** using multiple RNN architectures — from a basic SimpleRNN to a deep Bidirectional LSTM — with a model comparison.

---

## 🧠 What You'll Learn
- **SimpleRNN** — the baseline
- **Bidirectional LSTM** — reads sequences both forwards and backwards
- **Deep RNN** — stacking multiple LSTM layers
- Embedding layers (trained end-to-end)
- **EarlyStopping** + **ReduceLROnPlateau** callbacks
- Sequence padding and truncation

---

## 🚀 Run It

```bash
cd 05_RNNs_and_Sentiment
pip install -r requirements.txt

python demo.py --model bilstm           # best model
python demo.py --model simple_rnn       # baseline
python demo.py --model deep_rnn         # deep stacked
python demo.py --compare                # compare all three
```

---

## 📁 Structure

```
05_RNNs_and_Sentiment/
├── src/
│   └── rnn_sentiment.py   # Model builders + data loading
├── demo.py
├── results/
└── requirements.txt
```

---

## 📈 Results Comparison (IMDB Test Set, 5 epochs)

| Model | Test Accuracy |
|-------|---------------|
| SimpleRNN | ~85% |
| Deep RNN (3L) | ~86% |
| **BiLSTM** | **~88%** |

---

## 📈 Generated Plots

| Plot | Description |
|------|-------------|
| `simplernn_training.png` | Accuracy + loss curves |
| `bilstm_training.png` | Accuracy + loss curves |
| `deeprnn_*_training.png` | Accuracy + loss curves |
