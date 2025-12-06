# 🚢 Titanic Survivor Prediction — Machine Learning Project

This project predicts whether a passenger survived the Titanic disaster using machine learning approaches.  
Built entirely in a **Jupyter Notebook (`.ipynb`)**, the project focuses on **data preprocessing, feature encoding, scaling, ensemble models, and hyperparameter tuning**.

---

## 📁 Dataset

Source: **Kaggle – Titanic: Machine Learning from Disaster**

Files used:



---

## 📌 Features Used

- Pclass  
- Sex *(Label Encoded)*  
- Age  
- Fare *(MinMaxScaler Applied)*  
- SibSp  
- Parch  
- Embarked  

---

## 🧠 ML Techniques Used

| Component | Purpose |
|----------|----------|
| RandomForestClassifier | Primary model training |
| AdaBoostClassifier | Boosted final model |
| GridSearchCV / RandomizedSearchCV | Hyperparameter tuning |
| LabelEncoder | Categorical feature encoding |
| MinMaxScaler | Normalization/scaling |
| accuracy_score | Model evaluation metric |

**Final Accuracy:** `XX%` *(update with your actual value)*

---

## 🔄 Workflow

1. Loaded the dataset and explored feature distribution  
2. Cleaned dataset + handled missing values  
3. Encoded categorical features using **LabelEncoder**  
4. Scaled numerical values using **MinMaxScaler**  
5. Trained base model using **RandomForestClassifier**  
6. Improved model via **GridSearchCV / RandomizedSearchCV tuning**  
7. Boosted performance using **AdaBoostClassifier**  
8. Evaluated results using **accuracy_score**  
9. Generated and reviewed final predictions  

---

## 📊 Libraries Used

```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
pandas
numpy
