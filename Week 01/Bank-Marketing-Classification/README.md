# Bank Marketing – Term Deposit Subscription Prediction

A Machine Learning classification project that predicts whether a bank customer will subscribe to a term deposit based on demographic information, account details, and previous marketing campaign interactions.

The project focuses on data preprocessing, handling class imbalance, comparing classification algorithms, and optimizing the decision threshold to improve prediction performance.

## Project Overview

Banks use direct marketing campaigns to promote financial products such as term deposits. Identifying customers who are more likely to subscribe can help banks use their marketing resources more efficiently.

In this project, I developed and evaluated several machine learning models to predict whether a customer will subscribe to a term deposit (`yes` or `no`).

**Problem Type:** Binary Classification

**Dataset:** UCI Bank Marketing

**Target Variable:** `y`

## Dataset

The dataset was obtained from the UCI Machine Learning Repository.

- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Dataset file:** `bank-full.csv`
- **Number of records:** 45,211
- **Target classes:** `yes` and `no`

The dataset contains customer demographic information, bank account details, and marketing campaign data.

### Features

Some of the main features include:

| Feature | Description |
|---|---|
| age | Customer's age |
| job | Type of job |
| marital | Marital status |
| education | Education level |
| balance | Average yearly account balance |
| housing | Housing loan status |
| loan | Personal loan status |
| contact | Contact communication type |
| day | Last contact day of the month |
| month | Last contact month |
| campaign | Number of contacts during the current campaign |
| pdays | Days since the customer was last contacted |
| previous | Number of contacts before the current campaign |
| poutcome | Outcome of the previous marketing campaign |

The target variable `y` indicates whether the customer subscribed to a term deposit.

## Project Workflow

### 1. Data Exploration and Preprocessing

The dataset was explored to understand its structure, feature distributions, and missing values.

Preprocessing steps included:

- Checking missing values and data types.
- Handling missing categorical values by assigning the `unknown` category.
- Separating independent variables (`X`) and the target variable (`y`).
- Removing the `duration` feature.

**Why was `duration` removed?**

The `duration` feature represents the length of the last phone call. Since this information is only available after the conversation has ended, using it for a prediction made before contacting the customer would introduce data leakage.

Therefore, `duration` was excluded to make the prediction scenario more realistic.

### 2. Train, Validation and Test Split

The dataset was divided into three subsets using stratified sampling:

| Dataset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Test | 20% |

The validation set was used for model comparison and threshold optimization.

The test set was kept separate until the final evaluation.

### 3. Feature Engineering and Encoding

A preprocessing pipeline was created using Scikit-learn's `ColumnTransformer`.

- **StandardScaler:** Applied to numerical features.
- **OneHotEncoder:** Applied to categorical features.
- **handle_unknown="ignore":** Used to handle previously unseen categorical values.

The preprocessing transformations were fitted only on the training data to prevent data leakage.

### 4. Model Development

Three classification approaches were evaluated:

- Logistic Regression
- Balanced Logistic Regression (`class_weight="balanced"`)
- Random Forest Classifier (`n_estimators=200`, `class_weight="balanced"`)

The models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### 5. Model Comparison

Validation results using the default classification threshold:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.892 | 0.639 | 0.185 | 0.288 |
| Balanced Logistic Regression | 0.760 | 0.279 | 0.660 | 0.392 |
| Random Forest | 0.893 | 0.625 | 0.212 | 0.316 |

The results demonstrated the impact of class imbalance on classification performance.

Balanced Logistic Regression achieved higher recall, while Random Forest provided a different balance between precision and recall.

### 6. Feature Importance Analysis

Feature importance was examined using the Random Forest model.

The most influential features included:

| Feature | Importance |
|---|---:|
| balance | 0.1568 |
| age | 0.1345 |
| day_of_week | 0.1207 |
| campaign | 0.0676 |
| pdays | 0.0375 |
| poutcome_success | 0.0328 |
| contact_unknown | 0.0271 |
| previous | 0.0228 |

These values represent the relative importance assigned by the trained Random Forest model. They do not establish causal relationships.

### 7. Decision Threshold Optimization

The default classification threshold of 0.50 was not ideal for identifying the minority `yes` class.

Different thresholds were tested using the validation set.

| Threshold | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| 0.10 | 0.255 | 0.736 | 0.379 |
| 0.15 | 0.327 | 0.653 | 0.436 |
| 0.20 | 0.386 | 0.572 | 0.461 |
| 0.25 | 0.429 | 0.500 | **0.462** |
| 0.30 | 0.465 | 0.412 | 0.437 |

A threshold of **0.25** achieved the highest F1 Score among the tested values.

This optimization improved the balance between precision and recall compared with the default threshold.

## Final Model Evaluation

The final Random Forest model was trained using the combined training and validation datasets and evaluated on the previously untouched test set.

**Final Model:** Random Forest Classifier

**Decision Threshold:** 0.25

### Test Set Results

| Metric | Score |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.4374 |
| Recall | 0.4887 |
| F1 Score | 0.4616 |
| ROC-AUC | 0.7920 |

### Classification Report

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| No (0) | 0.93 | 0.92 | 0.92 |
| Yes (1) | 0.44 | 0.49 | 0.46 |

The final model achieved an F1 Score of approximately **0.462** and a ROC-AUC of **0.792** on the test set.

The validation and test F1 Scores were very close, suggesting that the model maintained similar performance on unseen data.

However, the model still has room for improvement, particularly in identifying customers belonging to the minority `yes` class.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
Bank-Marketing-Classification/
│
│── bank_marketing_classification.ipynb
│
├── images/
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── README.md
└── requirements.txt
```

*This is a suggested repository structure. File and folder names can be adjusted to match the final repository.*

## Future Improvements

Potential improvements for future development:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Experimenting with Gradient Boosting, XGBoost, and other ensemble methods.
- Applying cross-validation.
- Exploring additional feature engineering techniques.
- Evaluating probability calibration.
- Comparing different threshold selection strategies.
- Improving minority-class prediction performance.

## Key Learnings

Through this project, I practiced:

- Exploratory Data Analysis (EDA).
- Handling missing categorical values.
- Numerical scaling and categorical encoding.
- Train-validation-test splitting.
- Preventing data leakage.
- Classification model development and comparison.
- Handling imbalanced datasets.
- Feature importance analysis.
- Decision threshold optimization.
- Evaluating models on unseen test data.

## Author

**Ata Kalanyer**

Computer Programming Student | Aspiring AI & Machine Learning Engineer

- GitHub: [@atakalanyer](https://github.com/atakalanyer)
- LinkedIn: [Ata Kalanyer](https://www.linkedin.com/in/atakalanyer/)

---

*This project was developed as part of my journey in Machine Learning and Data Science.*