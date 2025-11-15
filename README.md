# 🎵 What Makes a Song a Hit? — Spotify Hit Prediction Using Machine Learning

**Aug 2025 – Sep 2025**  
**Author:** *Oskar Diyali*

This project analyzes what makes a song a **hit** by exploring **80,000+ Spotify tracks** using audio features, metadata, and lyrics. I built a full machine learning pipeline—from data cleaning to model evaluation—achieving **AUC > 0.86** and uncovering key traits such as a strong tempo cluster around **110–130 BPM**.

---

## 1. Project Overview

### Goal  
Predict whether a song becomes a “hit” using:

- Spotify audio features  
- Artist metadata  
- Billboard chart performance  
- Lyrics (TF–IDF transformed)

A track is labeled a **hit** if its Spotify popularity score is **≥ 45**.

### Why This Project Matters  
This project blends **music**, **data science**, and **machine learning**, turning subjective “hit intuition” into quantifiable patterns.

---

## 📁 2. Repository Structure
```
spotify-hit-prediction-ml/
│
├── 01_lyrics_feature_engineering.ipynb
├── 02_model_training_and_evaluation.ipynb
├── songs_cleaned_ready.csv    // Main Dataset for Modeling
├──
│
├── project_presentation_slides.pdf
├── requirements.txt
├── README.md
│
├── images/
│ ├── bpm_distribution.png
│ ├── feature_importance.png
│ ├── roc_curve.png
│ ├── confusion_matrix.png
│ └── lyrics_wordcloud.png
│
└── data/
└── README.md

```

---

## 🎼 3. Data Sources

This project uses the following public datasets:

- Spotify Tracks Dataset (Kaggle)  
- Spotify Million Song Dataset (Kaggle)  
- Billboard Hot Weekly Charts  
- Spotify Web API  
- Lyrics dataset (Millsong / Kaggle)  

**Final dataset sizes:**
- **80k+ audio + metadata records**
- **24k+ audio + lyrics records**

---

## 🛠 4. Feature Engineering

### 🔊 Audio Features  
- Danceability  
- Energy  
- Acousticness  
- Instrumentalness  
- Liveness  
- Loudness  
- Speechiness  
- Valence  
- Tempo (BPM)

### 🧩 Metadata Features  
- Artist popularity  
- Artist follower count  
- Album type  
- Release year/month  
- Explicit flag  
- Genre & label groups  

### ✍️ Lyrics (NLP)  
- Text normalization  
- TF–IDF vectorization  
- Word frequency analysis  
- Wordclouds  

### 🧪 Additional Engineering  
- One-hot encoding  
- Log-transformations  
- Handling imbalance  
- Multicollinearity reduction  

---

## 🤖 5. Modeling Approach

Models tested:

- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

### **Primary Evaluation Metrics**
- ROC–AUC  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### 🥇 **Best Model**
**Tuned XGBoost**  
- **AUC:** > 0.86  
- **Accuracy:** ~88%  
- Very stable across splits  

---

## 🔍 6. Key Findings

### 🎧 1. Tempo matters  
A strong hit cluster appears around **110–130 BPM**, especially near 120 BPM.

### 💃 2. Danceability & loudness predict hits  
High-energy, louder songs are more likely to chart.

### 🌐 3. Metadata is extremely predictive  
Artist popularity and follower count significantly improve model performance.

### 📝 4. Lyrics add interpretability  
TF–IDF reveals common emotional and thematic words in hit songs.

---

## 📓 7. Notebook Guide

### **`01_lyrics_feature_engineering.ipynb`**  
- Clean lyrics  
- TF–IDF vectorization  
- Merge audio + lyrics  
- Build lyric-based prediction features  

### **`02_model_training_and_evaluation.ipynb`**  
- Load all datasets  
- Clean & merge Spotify + Billboard + API sources  
- Feature engineering  
- Exploratory Data Analysis (EDA)  
- Train ML models (LogReg, RF, XGBoost, LightGBM, CatBoost)  
- Hyperparameter tuning  
- Generate ROC curves, confusion matrices, and accuracy metrics  

---

## 📈 8. Visualizations

(Add these after uploading your images to the `images/` folder)

```markdown
![BPM Distribution](images/bpm_distribution.png)
![Feature Importance](images/feature_importance.png)
![ROC Curve](images/roc_curve.png)
![Confusion Matrix](images/confusion_matrix.png)
![Lyrics Wordcloud](images/lyrics_wordcloud.png)
