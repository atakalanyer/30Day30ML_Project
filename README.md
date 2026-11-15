# 30Day30ML_Project
# Students Performance Prediction

A Machine Learning project that predicts students' mathematics exam scores using demographic information, parental education, lunch type, test preparation course, and reading and writing scores.

This project explores data preprocessing, exploratory data analysis, regression models, model evaluation, and hyperparameter tuning.

## Project Overview

The main objective is to predict students' mathematics scores (`math score`) using the Students Performance in Exams dataset.

**Dataset:** [Students Performance in Exams – Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

- 1,000 student records
- 5 categorical features
- 2 additional numerical features used as predictors
- 1 target variable: `math score`

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub

## Project Workflow

### 1. Data Collection
- Downloaded the dataset using KaggleHub.
- Loaded the CSV file using Pandas.

### 2. Exploratory Data Analysis (EDA)
- Inspected dataset structure and statistical summaries.
- Checked missing values.
- Examined numerical feature distributions.
- Investigated potential outliers using the IQR method.
- Analyzed relationships between variables using correlation analysis.

### 3. Data Preprocessing
- Applied Ordinal Encoding to parental education level.
- Applied One-Hot Encoding to categorical variables.
- Used StandardScaler for feature scaling.
- Split the dataset into training, validation, and test sets.

### 4. Model Training

Two regression algorithms were trained and evaluated:

- Linear Regression
- Random Forest Regressor

### 5. Model Evaluation

The models were evaluated using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### 6. Hyperparameter Optimization

GridSearchCV with 5-fold cross-validation was used to optimize the Random Forest model.

Best parameters:

```python
{
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 2
}
```

## Model Results

### Validation Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | **4.36** | **5.36** | **0.8673** |
| Random Forest | 4.89 | 6.05 | 0.8309 |
| Optimized Random Forest | 4.88 | 6.02 | 0.8326 |

### Final Test Results

**Best Performing Model: Linear Regression**

| Metric | Test Score |
|---|---:|
| MAE | 4.20 |
| MSE | 29.10 |
| RMSE | 5.39 |
| R² Score | **0.8804** |

Linear Regression achieved the best validation performance and was selected as the final model.

The final test R² score of 0.8804 indicates that the model explains approximately 88% of the variance in mathematics scores on the test set.

## Feature Analysis

Correlation analysis showed strong positive relationships between mathematics scores and:

- Reading Score: 0.8176
- Writing Score: 0.8026

Random Forest Feature Importance analysis also identified reading and writing scores as the most influential features.

| Feature | Importance |
|---|---:|
| Reading Score | 0.5353 |
| Writing Score | 0.2588 |
| Gender (Female) | 0.0619 |
| Gender (Male) | 0.0591 |

These values represent model-based feature importance, not causal relationships.

## Key Learnings

Through this project, I practiced:

- Exploratory Data Analysis
- Data Cleaning and Preprocessing
- Outlier Detection
- Categorical Encoding
- Feature Scaling
- Regression Algorithms
- Cross-Validation
- Hyperparameter Optimization
- Model Evaluation
- Feature Importance and Correlation Analysis

## Conclusion

This project demonstrated that a simpler model can outperform a more complex algorithm depending on the dataset.

Although Random Forest was optimized using GridSearchCV, Linear Regression achieved better results on the validation set and was selected for final testing.

This project is part of my **30 Days of Machine Learning** learning journey.

---

**Author:** Ata Kalanyer  
**GitHub:** [@atakalanyer](https://github.com/atakalanyer)
