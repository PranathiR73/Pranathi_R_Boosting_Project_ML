# 🚢 Titanic Survival Prediction Using Boosting (XGBoost)

An end-to-end machine learning classification project that analyzes historical passenger data, handles missing data anomalies, applies exploratory data analysis, and trains an advanced ensemble **Boosting** model to predict passenger survival outcomes.

---
## 📊 Dataset Description & Source Details

- **Dataset Title:** Titanic - Machine Learning from Disaster
- **Kaggle Source Link:** [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)
- Dataset used : train.csv
- **Dataset License & Origin:** Public domain dataset hosted on Kaggle, originally compiled from historical archives detailing the passenger manifests of the RMS Titanic's maiden voyage in 1912.
- **Dataset Structure & Dimensions:** 
  - **Rows:** 891 individual passenger records (training set).
  - **Columns:** 12 original features capturing demographic, socio-economic, and logistical attributes.
- **Target Variable:** 
  - `Survived`: Binary classification target (`0` = Did not survive, `1` = Survived).
- **Comprehensive Column Meanings & Feature Dictionary:**
  - `PassengerId`: Unique identification number assigned to each passenger. *(Dropped during cleaning)*
  - `Survived`: Target label indicating survival status (`0` or `1`).
  - `Pclass`: Ticket class representing socio-economic status (`1` = 1st Class / Upper, `2` = 2nd Class / Middle, `3` = 3rd Class / Lower).
  - `Name`: Full name of the passenger including title prefixes. *(Dropped during cleaning)*
  - `Sex`: Gender of the passenger (`male` or `female`).
  - `Age`: Age of the passenger in fractional or whole years.
  - `SibSp`: Number of siblings or spouses traveling aboard with the passenger.
  - `Parch`: Number of parents or children traveling aboard with the passenger.
  - `Ticket`: Unique alphanumeric ticket serial number. *(Dropped during cleaning)*
  - `Fare`: Monetary ticket price paid in British Pounds.
  - `Cabin`: Physical cabin number assigned to the passenger. *(Dropped due to >75% missing values)*
  - `Embarked`: Port of embarkation where the passenger boarded (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton).
## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description & Source](#dataset-description--source)
3. [Problem Statement & Objective](#problem-statement--objective)
4. [Data Quality & Cleaning Summary](#data-quality--cleaning-summary)
5. [Exploratory Data Analysis & Visualizations](#exploratory-data-analysis--visualizations)
6. [Model Approach & Selection](#model-approach--selection)
7. [Model Evaluation & Interpretation](#model-evaluation--interpretation)
8. [Conclusion, Limitations & Future Scope](#conclusion-limitations--future-scope)

---

## 1. Project Overview
This project follows an rigorous end-to-end data science pipeline. Using Python within a Google Colab notebook environment, it explores passenger demographics and ticket economics from the RMS Titanic disaster to predict survival with high accuracy using an optimized **XGBoost (Extreme Gradient Boosting)** classifier.

---

## 2. Dataset Description & Source
* **Dataset Title:** Titanic - Machine Learning from Disaster
* **Kaggle Source Link:** [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)
* **Dataset Structure:** Contains **891 rows** (passengers) and **12 columns** (features capturing demographics, ticket class, fare, family counts, and survival status).
* **Target Variable:** `Survived` (Binary: `0` = Did not survive, `1` = Survived).

---

## 3. Problem Statement & Objective
* **Real-World Problem:** Maritime disasters involve catastrophic loss of life. Historically, emergency protocols and resource allocations (such as lifeboats) lacked data-driven prioritization.
* **Why It Matters:** Understanding historical survival drivers helps establish modern safety standards and highlights systemic rescue disparities during crises.
* **Project Objective:** Build a robust binary classification model using a **Boosting** algorithm to accurately predict passenger survival based on individual traits and ticket features.

---

## 4. Data Quality & Cleaning Summary
* **Missing Values Check:** Identified missing entries in `Age` (177 rows), `Cabin` (687 rows), and `Embarked` (2 rows). Zero duplicate rows were found.
* **Data Cleaning Decisions:**
  1. **Age Imputation:** Filled missing `Age` values using the dataset's **median age** to prevent outlier skew.
  2. **Embarked Imputation:** Filled missing port values with the **mode** (Southampton).
  3. **Dropping Sparse Features:** Removed the `Cabin` column as over 75% of rows lacked records.
  4. **Removing Unnecessary Identifiers:** Dropped `PassengerId`, `Name`, and `Ticket` to prevent model overfitting.
  5. **Categorical Encoding:** Mapped string values in `Sex` (`male`/`female`) and `Embarked` (`S`/`C`/`Q`) into numerical integers.

---

## 5. Exploratory Data Analysis & Visualizations
The notebook generates four key visualizations with analytical observations:
1. **Passenger Survival Distribution (Target):** Highlights the class imbalance between survivors and casualties.
2. **Survival Rate by Gender:** Proves that female passengers had a vastly higher survival probability, validating rescue protocols.
3. **Passenger Class Impact (`Pclass`):** Shows that 1st-class passengers experienced higher survival rates compared to 3rd-class passengers due to deck placement and evacuation priority.
4. **Age Distribution Stacked by Survival:** Reveals that young children had higher relative survival rates, while adults aged 20–40 accounted for the highest volume of casualties.

---

## 6. Model Approach & Selection
* **Task Type:** Supervised Machine Learning – **Binary Classification**.
* **Chosen Algorithm:** **XGBoost Classifier (Extreme Gradient Boosting)**.
* **Justification:** Boosting builds an ensemble of decision trees sequentially, where each new tree focuses specifically on correcting the residual errors of prior trees. This iterative boosting mechanism excels at handling structured tabular data with exceptional predictive accuracy.

---

## 7. Model Evaluation & Interpretation
* **Train-Test Split:** 80% training set, 20% testing set (stratified).
* **Performance Metrics:**
  * **Model Accuracy Score:** **79.89%**
  * **Precision & Recall:** Balanced successfully across both survival classes via a Classification Report and Confusion Matrix.
* **Simple Interpretation:** Accuracy measures the percentage of correct predictions made on unseen test data, proving that the boosting model effectively captures non-linear feature interactions between passenger class, gender, and ticket fare.

---

## 8. Conclusion, Limitations & Future Scope
* **Key Findings:** Gender and socio-economic class (`Pclass`) are the single most powerful predictors of survival during maritime emergencies.
* **Limitations:** The historical dataset lacks granular real-time context, such as cabin proximity to lifeboats or crew assistance logs.
* **Future Improvements:** 
  * Perform hyperparameter tuning using `GridSearchCV` or `Optuna` to optimize learning rates and tree depth.
  * Extract title prefixes (`Mr`, `Mrs`, `Miss`, `Master`) from passenger names to engineer advanced social status features.
