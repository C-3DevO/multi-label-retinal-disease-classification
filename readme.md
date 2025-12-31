# Beyond Baselines: Transfer Learning for Multi-Label Retinal Disease Classification

## Project Overview
This project investigates **transfer learning strategies for multi-label retinal disease classification** using deep learning. The task focuses on detecting **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Age-related Macular Degeneration (AMD)** from fundus images under **limited annotated data conditions**.

Using pretrained CNN backbones and transformer architectures, we systematically evaluate:
- Transfer learning strategies
- Loss functions for class imbalance
- Attention mechanisms
- Explainable AI (Grad-CAM)
- Ensemble learning

All experiments are conducted on the **ODIR retinal dataset**, following the official course instructions at the University of Oulu.

---

## Motivation
Medical image annotation is expensive, slow, and requires expert knowledge. Transfer learning enables models pretrained on large datasets to generalize effectively to small medical datasets. However, **model adaptation strategy**, **loss design**, and **attention mechanisms** significantly affect performance—especially for **minority disease classes**.

This project aims to **surpass baseline performance** while maintaining interpretability, robustness, and reproducibility.

---

## Dataset
**Dataset:** ODIR (Ocular Disease Intelligent Recognition)

| Split | Images |
|------|--------|
| Training | 800 |
| Validation | 200 |
| Offsite Test | 300 |
| Onsite Test | 250 |

- Image resolution: **256 × 256**
- Normalization: **ImageNet statistics**
- Class imbalance handled using **loss re-weighting** (no resampling)

---

## Tasks Completed

### Task 1: Transfer Learning
- No fine-tuning (baseline)
- Classifier-only fine-tuning
- Full fine-tuning

**Baseline onsite F1-scores (as required):**
- EfficientNet: **60.4**
- ResNet18: **56.7**

---

### Task 2: Loss Functions
- Binary Cross-Entropy (BCE)
- Focal Loss
- Class-Balanced Loss

Key findings:
- **Focal Loss** significantly improves AMD detection
- **Class-Balanced Loss** yields the largest relative gain for ResNet18 minority classes

---

### Task 3: Attention Mechanisms
- Squeeze-and-Excitation (SE)
- Multi-Head Attention (MHA)

Results:
- SE + BCE achieves the **highest overall onsite F1-score**
- MHA improves **minority-class sensitivity**, particularly AMD

---

### Task 4: Open Questions
- Why ensemble learning does not always outperform the strongest single model
- Trade-offs between sensitivity and generalization
- Impact of dataset size on transformer effectiveness

---

### Task 5: Technique Report
A full technical report is included describing:
- Methodology and experimental setup
- Quantitative and qualitative results
- Loss and attention analysis
- Grad-CAM interpretability
- Ensemble behavior
- Limitations and future work

---

## Model Architectures
- **CNNs:** EfficientNet, ResNet18
- **Attention:** Squeeze-and-Excitation (SE), Multi-Head Attention (MHA)
- **Transformers:** Vision Transformer (ViT), Swin Transformer

---

## Explainable AI
**Grad-CAM** is used to visualize class-specific activation maps:
- Confirms focus on clinically relevant retinal regions
- Reduces reliance on background artifacts
- Improves trust and interpretability of predictions

---

## Folder Structure
```
.
├── pretrained_backbone/
├── source_code/
│   ├── OulunOwls_task1_1_effnet.py
│   ├── OulunOwls_task1_1_resnet.py
│   ├── ...
│   └── OulunOwls_task4_1_transformer.py
├── trained_models/
│   ├── OulunOwls_task1_1_effnet.pt
│   ├── OulunOwls_task2_2_resnet.pt
│   ├── best_efficientnet.pt
│   ├── best_resnet18.pt
│   └── ...
├── code_template.py
├── README.md
└── OulunOwls_Technical_Report.pdf
```



## Setup and Reproducibility

### 1. Dataset Download (Required)
The dataset used in this project is provided via **Kaggle** as part of the course final project.

1. Create a Kaggle account (if you do not already have one).
2. Download the dataset from:
   https://www.kaggle.com/competitions/final-project-deep-learning-fall-2025/data
3. Extract the downloaded files into a local directory
4. Update the dataset paths inside the training scripts if needed.

> **Note:** The onsite test labels are hidden and are used only for online evaluation.

---

### 2. Environment Setup
We recommend using a virtual environment to ensure reproducibility.

python -m venv dl_env
source dl_env/bin/activate   # Linux / macOS
# dl_env\Scripts\activate  # Windows

pip install torch torchvision numpy pandas pillow scikit-learn matplotlib

---

### 3. Training and Evaluation

### 1. Environment Setup
```bash
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib
```

### 2. Training Example
```bash
python source_code/OulunOwls_task2_2_effnet.py
```

### 3. Evaluation
Each script automatically evaluates:
- Precision
- Recall
- F1-score (per class & average)
- Cohen’s Kappa

Thresholds are configurable inside each script.

---

## Key Results (Onsite Test)

| Method | Avg F1 |
|------|-------|
| EfficientNet Baseline | 60.4 |
| ResNet18 Baseline | 56.7 |
| Full Fine-Tuned EfficientNet | 80.12 |
| EfficientNet + Focal Loss | 82.55 |
| EfficientNet + SE Attention | **82.10** |
| Weighted Ensemble | 79.15 |

---

## Key Insights
- Fine-tuning depth matters more than architecture scaling
- Loss design is critical for minority disease detection
- Attention mechanisms improve feature discrimination
- Ensemble learning improves robustness but not always peak performance
- Transformer models require larger datasets to outperform CNNs

---

## Conclusion
This project demonstrates that **carefully designed transfer learning pipelines**—combining fine-tuning strategies, imbalance-aware loss functions, and attention mechanisms—can significantly outperform pretrained baselines in multi-label medical image classification.

The methodology and insights presented here are extensible to other **small-scale medical imaging problems**.

---

## Authors
**Brian Kiprop Kibor**  
**Md Saddam Hossain**  
**Remus Shamim Ahamed**  

University of Oulu  

📧 brian.kibor@student.oulu.fi

