# Audio Deepfake Detection Dashboard

© 2026 Mohd Aimann. All rights reserved.

This project is a Final Year Project (FYP) dashboard for audio deepfake detection using machine learning techniques. The system displays experiment results and allows users to upload feature-based CSV datasets to classify audio samples as real or fake.

---

## Project Overview

Audio deepfake technology can generate or manipulate speech to sound like real human voices. This creates risks in cybersecurity, digital forensics, misinformation, and voice authentication.

This project focuses on detecting fake audio using handcrafted and learning-driven audio features. The proposed framework compares three feature representations:

- MFCC
- WavLM
- Integrated MFCC + WavLM

The framework also compares three machine learning classifiers:

- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## Main Features

- User registration and login
- Admin and user dashboard separation
- User dataset upload using feature CSV
- Prediction of real or fake audio samples
- Downloadable prediction results
- User result history
- Admin monitoring of user submissions
- Experiment dashboard for FYP results
- Within-dataset and cross-dataset evaluation results
- Confusion matrix display
- Dataset overview and feature extraction summary

---

## Evaluation Approach

Two evaluation strategies were used in this project.

### Within-Dataset Evaluation

The model was trained and tested using the ASVspoof2019 LA dataset.

### Cross-Dataset Evaluation

The model was trained using ASVspoof2019 LA and tested using the Fake-or-Real dataset. This was used to evaluate the generalization ability of the detection models.

---

## Dataset Summary

| Dataset | Audio Type / Category | Real / Bonafide Samples | Fake / Spoof Samples | Total Samples |
|---|---|---:|---:|---:|
| ASVspoof2019 LA | Bonafide + Logical Access spoof audio | 500 | 500 | 1000 |
| Fake-or-Real | Real + TTS-generated fake audio | 500 | 500 | 1000 |

---

## Feature Extraction

| Feature Type | Number of Features | Description |
|---|---:|---|
| MFCC | 80 | 40 MFCC mean values and 40 MFCC standard deviation values |
| WavLM | 768 | Speech embeddings extracted from a pre-trained WavLM model |
| MFCC + WavLM | 848 | Concatenation of MFCC and WavLM features |

---

## Technologies Used

- Python
- Streamlit
- Pandas
- SQLite
- Scikit-learn
- XGBoost
- Joblib
- Pillow
