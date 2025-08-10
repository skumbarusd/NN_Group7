# 🎼 Composer Prediction using MIDI Classic Music Dataset

##  Project Overview
This project aims to predict the composer of a classical music piece using MIDI file data. By analyzing musical features extracted from MIDI files, a machine learning model is trained to classify pieces according to their composers.

The goal is to explore how computational methods can identify stylistic patterns in classical music and apply them to composer recognition.

---

## 📂 Dataset
- **Source:** [MIDI Classic Music Dataset - Kaggle](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)
- **Description:** The dataset contains MIDI files of classical compositions from well-known composers such as Bach, Beethoven, Chopin, and more.
- **Contents:**  
  - `composer`: The composer’s name (target label)
  - MIDI file features extracted from pitch, duration, tempo, and note patterns

---

## ⚙️ Project Workflow
1. **Data Acquisition**  
   Download dataset from Kaggle and extract MIDI files.
   
2. **Feature Extraction**  
   - Parse MIDI files using `mido` or `pretty_midi`
   - Extract features like:
     - Note pitch distributions
     - Note duration patterns
     - Tempo variations
     - Chord progressions

3. **Data Preprocessing**  
   - Handle missing values
   - Normalize numerical features
   - Encode categorical variables

4. **Model Training**  
   - Train classification models such as:
     - Random Forest
     - XGBoost
     - Neural Networks
   - Evaluate using accuracy, F1-score, and confusion matrix

5. **Model Evaluation & Tuning**  
   - Hyperparameter tuning with Grid Search / Random Search
   - Cross-validation to prevent overfitting

6. **Prediction**  
   - Input a new MIDI file → Extract features → Predict composer

---

## 🛠️ Technologies Used
- **Languages:** Python
- **Libraries:**
  - `pandas`, `numpy` – Data handling
  - `mido`, `pretty_midi` – MIDI parsing
  - `scikit-learn` – Machine learning models & preprocessing
  - `matplotlib`, `seaborn` – Visualization
  - `xgboost`, `lightgbm` – Advanced ML models
- **Environment:** Jupyter Notebook

---

## 📊 Example Results
- Best model accuracy: `XX%` (replace with actual result from notebook)
- Confusion matrix visualizations show strong performance on most composers.

---

## 🚀 How to Run
1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/composer-prediction.git
   cd composer-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset from Kaggle**
   ```bash
   kaggle datasets download -d blanderbuss/midi-classic-music
   unzip midi-classic-music.zip -d data/
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook FinalProject_Group7.ipynb
   ```

---

## 📌 Future Work
- Expand dataset with more composers
- Try deep learning models like LSTMs or Transformers for sequence modeling
- Build a web app to predict composers from uploaded MIDI files

---
