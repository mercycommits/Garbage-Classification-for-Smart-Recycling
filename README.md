# 🗑️ Garbage Classification with EfficientNetB0

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

Recycling systems often rely on manual sorting, which is slow, inconsistent, and expensive. 
This project explores how deep learning can help automate waste classification by training an EfficientNetB0 model to identify six types of garbage.
With over 93% test accuracy, the model shows that even small, efficient neural networks can significantly support smarter and more sustainable waste-management solutions.

---

## 📊 Results

### **Overall Metrics**
- **Test Accuracy:** 93.7%  
- **Macro F1-Score:** 93%

### **Class Performance**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| cardboard | 0.99 | 0.99 | 0.99 |
| glass     | 0.91 | 0.93 | 0.92 |
| metal     | 0.93 | 0.96 | 0.95 |
| paper     | 0.94 | 0.96 | 0.95 |
| plastic   | 0.90 | 0.90 | 0.90 |
| trash     | 0.97 | 0.84 | 0.90 |

---

## 🚀 Quick Start

### **1. Install dependencies**
```bash
pip install -r requirements.txt
```
### **2. Dataset Setup**
Download TrashNet
- Option 1 : via Kaggle API
  ```bash
  import kagglehub
  path = kagglehub.dataset_download("feyzazkefe/trashnet")
  print("Path to dataset files:", path)
- Option 2 : Manually from  [Kaggle TrashNet Dataset](https://www.kaggle.com/datasets/feyzazkefe/trashnet)
### **3. Run Notebooks**
- 1_eda_data_augmentation.ipynb — EDA + Augmentation
- 2_model_training_evaluation.ipynb — Training + Evaluation
---
## 🧠 Model Architecture
- **Backbone:** EfficientNetB0 (ImageNet pretrained)

- **Fine-tuning:** Last 20 layers unfrozen

- **Head:** GlobalAveragePooling2D → Dense(6)

- **Regularization:** Dropout(0.4)

- **Optimizer:** Adam (learning rate 1e-4)
---
## ⚡ Key Features
- Handles class imbalance with strategic augmentation

- Transfer Learning with EfficientNetB0

- Comprehensive evaluation with confusion matrix

- Data augmentation for better generalization

- Class weights for minority classes
---
## 📁 Dataset Info
- Original images: 2,527

- After augmentation: 3,063

- Categories: 6

- Image size: 512 × 384
---
## 🛠️ Tech Stack
- TensorFlow / Keras

- OpenCV

- Scikit-learn

- Matplotlib / Seaborn
---


