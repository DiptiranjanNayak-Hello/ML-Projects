# 🏡 California House Price Prediction — Machine Learning Project

This project predicts the median house value in Californian districts using regression techniques. It demonstrates a complete machine learning pipeline, from raw data cleaning to advanced model tuning.

Built entirely in a **Jupyter Notebook (`.ipynb`)**, the project focuses on: **Data Transformation (Log), Feature Engineering, One-Hot Encoding, Feature Scaling, Ensemble Modeling, and Hyperparameter Tuning.**

---

## 📁 Dataset

Source: **California Housing Dataset** (often found in Scikit-learn or similar public domain repositories).

| File | Description |
| :--- | :--- |
| `housing.csv` | Contains geographical and socioeconomic features for Californian districts. |

---

## 📌 Features & Preprocessing Applied

The raw data was transformed extensively to improve model performance:

| Feature/Action | Technique Used | Purpose |
| :--- | :--- | :--- |
| `total_rooms`, `total_bedrooms`, `population`, `households` | **Log Transformation** (`np.log(x+1)`) | Normalized highly skewed distributions. [Image of log transformation effect on skewed data] |
| `ocean_proximity` | **One-Hot Encoding** (`pd.get_dummies`) | Converted categorical location data into numerical features. |
| **New Feature: `bedroom_ratio`** | **Feature Engineering** (`total_bedrooms / total_rooms`) | Calculated the ratio of bedrooms to total rooms, a strong predictor of house quality. |
| **New Feature: `household_rooms`** | **Feature Engineering** (`total_rooms / households`) | Calculated the average rooms per household (density/space). |
| All numerical features | **StandardScaler** | Standardized features to have zero mean and unit variance for model consumption. |

---

## 🧠 ML Techniques & Results

The project tested two models and optimized the best performer using cross-validation. The final metric reported is the **R-squared score ($R^2$)**, which indicates the model's predictive accuracy.

| Component | Technique/Role | Result ($R^2$ Score) |
| :--- | :--- | :--- |
| **Linear Regression** | Baseline Model | $0.62$ (Approximate) |
| **RandomForestRegressor** | Ensemble Model | $0.78$ (Approximate) |
| **GridSearchCV** | Hyperparameter Tuning | Optimized $\text{n\_estimators}$, $\text{min\_samples\_split}$, $\text{max\_depth}$ over 5 folds. [Image of the GridSearchCV process for hyperparameter tuning] |
| **Best Model ($\text{RandomForestRegressor}$)** | Final Tuned Model | $\mathbf{0.81436}$ |

The final tuned Random Forest model explains approximately $\mathbf{81.4\%}$ of the variance in median house values.

---

## 🔄 Workflow

1.  🧹 **Data Cleaning:** Loaded data and handled missing values using `dropna()`.
2.  🎯 **Data Splitting:** Divided data into $\text{train}$ and $\text{test}$ sets using a $\mathbf{80/20}$ ratio.
3.  📐 **Transformation & Encoding:** Applied **Log Transformation** and **One-Hot Encoding** as detailed above.
4.  🛠️ **Feature Engineering:** Calculated and added the $\text{bedroom\_ratio}$ and $\text{household\_rooms}$ features.
5.  📏 **Scaling:** Applied $\text{StandardScaler}$ to all numerical features.
6.  📈 **Modeling:** Trained **Linear Regression** and **RandomForestRegressor**.
7.  ⚙️ **Tuning:** Used **GridSearchCV** with the parameter space `{'n_estimators' : [100, 200, 300], 'min_samples_split' : [2, 4], 'max_depth' : [None, 4, 8]}` to find the best Random Forest configuration.
8.  ✅ **Evaluation:** Tested the final $\text{best\_estimator\_}$ on the held-out test set, achieving the final $R^2$ score.

---

## 💻 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor