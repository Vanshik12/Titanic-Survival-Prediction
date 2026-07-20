# 🚢 Titanic Survival Prediction

An end-to-end machine learning project predicting Titanic passenger survival using demographic and travel data, built in Google Colab and submitted to Kaggle's official leaderboard.

## 📌 Objective

Predict whether a passenger survived the Titanic disaster based on features like passenger class, sex, age, fare, and family size — and identify which factors influenced survival the most.

## 📊 Dataset

- **Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)
- **Size:** 891 training passengers, 12 original features
- **Target variable:** `Survived` (0 = Died, 1 = Survived)

## 🔧 Methodology

1. **Exploratory Data Analysis** — survival patterns by gender, passenger class, age, and embarkation port
2. **Data Cleaning** — handled missing values in `Age`, `Embarked`, and `Cabin`; encoded categorical variables
3. **Feature Engineering** — `FamilySize`, `IsAlone`, age and fare groupings for interpretability
4. **Model Development** — trained and compared three models:
   - Logistic Regression (with feature scaling via Pipeline)
   - Random Forest
   - XGBoost
5. **Model Evaluation** — 5-fold cross-validation, confusion matrix, ROC-AUC, feature importance
6. **Hyperparameter Tuning** — GridSearchCV for optimal Random Forest parameters
7. **Kaggle Submission** — generated predictions on the official test set

## 📈 Results

| Model | Cross-Validated Accuracy |
|---|---|
| Logistic Regression |  80.45% |
| Random Forest | 81.01% |
| XGBoost | 79.33% |

**Best model:** _add model name_
**Kaggle Leaderboard Score:** **0.77**

**Key insight:** Gender and passenger class were the strongest predictors of survival, consistent with the historical "women and children first" evacuation protocol and lifeboat proximity for upper-class cabins.

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `Google Colab`

## 🚀 How to Run

1. Download `train.csv` and `test.csv` from the [Kaggle Titanic page](https://www.kaggle.com/competitions/titanic/data)
2. Open `Titanic_Survival_Prediction.ipynb` in Google Colab
3. Run all cells (`Runtime → Run all`) — you'll be prompted to upload both CSV files in the first cell
4. `submission.csv` is generated automatically at the end, ready for Kaggle submission

## 📁 Repository Structure
