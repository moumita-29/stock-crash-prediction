# Stock Market Crash Prediction using Deep Learning

## Overview

This project builds a deep learning model to predict short-term stock market crashes using historical financial data and technical indicators. The system uses a Long Short-Term Memory (LSTM) neural network with an attention mechanism to learn temporal patterns in financial time series.

The goal is to detect potential market crashes several days in advance using past market behavior.

---

## Dataset

Financial market data is collected using the Yahoo Finance API.

**Assets used**

* NIFTY 50 Index
* Bank Nifty
* Reliance Industries
* TCS
* Infosys
* HDFC Bank
* ICICI Bank
* SBI
* Larsen & Toubro
* ITC

**Time Period**

2000 – Present

**Dataset Statistics**

* Total sequences: ~58,000
* Sequence length: 30 days
* Features per timestep: 14

---

## Feature Engineering

Several technical indicators and statistical features are generated from the raw market data.

**Technical Indicators**

* Relative Strength Index (RSI)
* Exponential Moving Average (EMA20)
* Exponential Moving Average (EMA50)
* Bollinger Bands

**Statistical Features**

* Rolling volatility
* Future returns
* Crash labels

A crash is defined as a **price drop greater than 3% within the next 5 trading days**.

---

## Model Architecture

The model is implemented in **PyTorch** and consists of:

* LSTM layer for time series learning
* Attention mechanism for important timestep weighting
* Fully connected layer for binary classification

Loss function:

Binary Cross Entropy with Logits (BCEWithLogitsLoss) with class weighting to handle crash imbalance.

---

## Project Pipeline

The complete machine learning pipeline consists of the following stages:

1. Data collection from Yahoo Finance
2. Feature engineering and technical indicator computation
3. Time-series dataset generation
4. LSTM model training
5. Model evaluation and crash prediction

Pipeline scripts:

```
download_data.py
build_features.py
create_dataset.py
train.py
evaluate.py
```

---

## Results

Model performance on the test dataset:

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | ~78%  |
| Crash Precision | ~34%  |
| Crash Recall    | ~35%  |
| F1 Score        | ~0.34 |

The model successfully detects a portion of market crashes while maintaining reasonable accuracy on normal market conditions.

---

## Project Structure

```
stock-crash-prediction
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   ├── crash_model.pth
│   └── scaler.pkl
│
├── src
│   ├── download_data.py
│   ├── build_features.py
│   ├── create_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── model.py
│
├── notebooks
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/stock-crash-prediction.git
cd stock-crash-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Pipeline

Step 1 — Download market data

```
python src/download_data.py
```

Step 2 — Build features

```
python src/build_features.py
```

Step 3 — Create dataset

```
python src/create_dataset.py
```

Step 4 — Train the model

```
python src/train.py
```

Step 5 — Evaluate model performance

```
python src/evaluate.py
```

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Yahoo Finance API
* Pandas-TA

---

## Future Improvements

* Add macroeconomic indicators
* Use Transformer-based models for time series
* Improve crash detection using additional market signals
* Deploy model as a real-time prediction API

---

## Author

Moumita Paul
