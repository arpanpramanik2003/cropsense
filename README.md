# 🌿 CropSense: Quality Detection in Solanaceous Crops Using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

> A novel multi-headed CNN architecture for simultaneous classification of potato and tomato crops with quality assessment using explainable AI.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Explainability](#model-explainability)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [Contributors](#contributors)

## 🎯 Overview

CropSense is a deep learning-based system that performs **dual-task classification**:
1. **Crop Classification**: Distinguishes between potato and tomato crops
2. **Quality Assessment**: Identifies healthy vs. diseased samples

Traditional agricultural inspection methods are time-consuming and error-prone. This project bridges the gap between crop identification and quality control using a unified multi-task learning framework with **99.9% crop classification accuracy** and **98.5% quality assessment accuracy**[attached_file:1].

### Problem Statement
Manual crop inspection by experts is:
- Time-consuming and labor-intensive
- Prone to human error
- Not scalable for large-scale agriculture

Most existing automated solutions focus on either classification OR quality assessment in isolation, missing opportunities for shared feature learning[attached_file:1].

## ✨ Key Features

- **Multi-Task Learning**: Shared feature extractor with task-specific heads
- **Lightweight Architecture**: 30% parameter reduction compared to separate models
- **Explainable AI**: Grad-CAM visualizations for transparent decision-making
- **High Performance**: 
  - 99.9% accuracy in crop classification (potato vs. tomato)
  - 98.5% accuracy in quality assessment (healthy vs. diseased)
- **Production-Ready**: Streamlit web interface for real-time predictions
- **Robust Data Augmentation**: Geometric and photometric transformations
- **Fast Training**: Optimized for NVIDIA P100 GPU (26 minutes/50 epochs)


### Optimization Framework

- **Loss Function**: Unweighted sum of cross-entropy losses
- **Optimizer**: Adam with cyclic learning rate (1e-5 to 1e-3)
- **Regularization**: 
- L2 weight decay (λ = 0.01)
- 50% dropout
- Batch normalization
- **Training Protocol**:
- Early stopping (patience=7)
- Learning rate reduction on plateau (factor=0.2, patience=2)
- Max 50 epochs, batch size 32

## 📊 Dataset

### Dataset Composition
Custom balanced dataset of **10,000 images** (256×256 resolution):

| Category | Healthy | Diseased | Total |
|----------|---------|----------|-------|
| Potato | 2,500 | 2,500 | 5,000 |
| Tomato | 2,500 | 2,500 | 5,000 |
| **Total** | **5,000** | **5,000** | **10,000** |

### Data Sources
1. **Tomato Dataset** (IEEE Access): 7,226 images across 4 states (Unripe, Ripe, Old, Damaged)
 - Old/Damaged → Diseased
 - Unripe/Ripe → Healthy
 
2. **Potato Dataset**: 200,000+ images covering diseases like Common Scab, Dry Rot, Gangrene, Violet Root Rot[attached_file:1]

### Data Split
- **Training Set**: 8,000 images (80%)
- **Testing Set**: 2,000 images (20%)

### Augmentation Techniques
- Rotation
- Horizontal/Vertical flipping
- Brightness adjustments
- Zoom/Shift transformations

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)


### Clone Repository
git clone "https://github.com/arpanpramanik2003/cropsense.git"
cd cropsense


### Install Dependencies
- pip install -r requirements.txt

  
**Output**:
- Crop Prediction: tomato (99.91% confidence)
- Condition Prediction: Disease (99.40% confidence)


### Streamlit Web Application
- streamlit run app.py


Access the interface at `http://localhost:8501`

**Features**:
- Drag-and-drop image upload
- Real-time predictions
- Grad-CAM visualizations
- Batch processing (up to 20 images/sec)
- Confidence threshold filtering (>90%)

## 📈 Results

### Classification Performance

#### Crop Classification (Test Set)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Potato | 1.00 | 1.00 | 1.00 | 1000 |
| Tomato | 1.00 | 1.00 | 1.00 | 1000 |
| **Accuracy** | | | **1.00** | **2000** |

#### Quality Assessment (Test Set)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Disease | 0.99 | 0.98 | 0.98 | 1000 |
| Healthy | 0.98 | 0.99 | 0.99 | 1000 |
| **Accuracy** | | | **0.98** | **2000** |

### Training Metrics
- **Crop Classification Loss**: 0.0037 (train), 0.0088 (validation)
- **Quality Assessment Loss**: 0.0457 (train), 0.0401 (validation)
- **Training Time**: 26 minutes on NVIDIA P100 GPU
- **Convergence**: Optimal at epoch 18 (crop), epoch 20 (quality)[attached_file:1]

### Confusion Matrix Analysis
- **Crop Classification**: Only 2 misclassifications out of 2000 samples (0.1% error)
- **Quality Assessment**: 30 total errors (17 false healthy, 13 false diseased)[attached_file:1]

## 🔍 Model Explainability

### Grad-CAM Visualization

We implement **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations for both tasks:

#### Mathematical Foundation
For each task head \( t \in \{\text{crop}, \text{condition}\} \):

\[
\alpha_{k}^{t,c} = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y_{c}^{t}}{\partial A_{ij}^{k}}
\]

\[
L_{t}^{c} = \text{ReLU}\left(\sum_{k} \alpha_{k}^{t,c} A^{k}\right)
\]

Where:
- \( A^{k} \): Activations from last Conv2D layer (128-channel)
- \( y_{c}^{t} \): Class score for task \( t \) and class \( c \)
- \( Z \): Normalization factor (width × height)

#### Key Insights (200 validation samples)
**Crop Classification**:
- Potato: Focuses on skin texture and shape contours (α = 0.79 ± 0.12)
- Tomato: Highlights stem attachment and surface gloss

**Quality Assessment**:
- Healthy: Diffuse activations (mean σ = 0.18)
- Diseased: Localized at lesion sites (89% correlation with visible defects)[attached_file:1]

## 🌐 Deployment

### Streamlit Interface Features
- **Image Upload**: Drag-and-drop or browse (JPG, JPEG, PNG)
- **Real-Time Processing**: Instant predictions with confidence scores
- **Dual Heatmaps**: Separate Grad-CAM visualizations for crop and quality tasks
- **Batch Processing**: Handle multiple images simultaneously
- **Quality Thresholds**: Filter predictions below 90% confidence


## 🔮 Future Work

### Dataset Expansion
- Incorporate 5,000+ field-captured images with natural occlusions
- Add temporal growth stage variations
- Include multiple disease severity levels

### Model Optimization
- **Quantization**: Target <50MB for Raspberry Pi deployment
- **Acceleration**: Implement TensorRT for real-time processing
- **Compression**: Knowledge distillation for edge devices

### Application Enhancement
- **Disease Severity Quantification**: Mild/Moderate/Severe classification
- **Multilingual Support**: Regional language interfaces for farmers
- **Mobile App**: Android/iOS deployment
- **Multi-Crop Extension**: Expand to other vegetable families
- **Multispectral Integration**: Incorporate NIR/thermal imaging

## 👥 Contributors

- **Shibdas Dutta** - Dept. of CSE, Amity University Madhya Pradesh  
  📧 shibdas.dutta@gmail.com

- **Arpan Pramanik** - Dept. of CSE, The Neotia University  
  📧 pramanikarpan089@gmail.com

- **Diya Chanda** - Dept. of CSE, The Neotia University  
  📧 chandasujata01@gmail.com
