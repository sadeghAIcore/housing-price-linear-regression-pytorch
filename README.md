# Housing Price Prediction 🏠
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

A Deep Learning-based regression project to predict house prices in California using **PyTorch**. This project implements a full machine learning pipeline, including data preprocessing, feature scaling, and GPU-accelerated training.

## 📋 Project Overview
The goal is to estimate the median house value based on 8 demographic and geographic features. The model utilizes a Neural Network architecture to capture relationships within the California Housing dataset.

## 🏗 Project Structure
The repository is organized as follows:
```text
├── data/               # Raw dataset (california_housing.csv)
├── models/             # Saved model weights (.pth files)
├── notebooks/          # Data analysis and visual demonstrations
├── src/
│   ├── model.py        # Model architecture
│   ├── train.py        # Training loop and optimization logic
│   └── predict.py      # Inference script for new data
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
