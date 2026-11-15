import os
import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
print("İndirilen klasör yolu:", path)

csv_dosyasi = os.path.join(path, "StudentsPerformance.csv")
df = pd.read_csv(csv_dosyasi)

#Veri Setini İnceleme
print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

#Sayisal sütunları tespit edip aykırı değerleri çıkartma işlemi.
sayisal_sutunlar = ["math score", "reading score", "writing score"]

df_kopya = df.copy()

aykiri_deger_maskesi = False

for sutun in sayisal_sutunlar:
    q1 = df_kopya[sutun].quantile(0.25)
    q3 = df_kopya[sutun].quantile(0.75)

    iqr = q3-q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df_kopya[sutun] < alt_sinir) | (df_kopya[sutun] > ust_sinir)
    )
    
    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi
    
    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")
    
    if sutun_maskesi.any():
        print(f"Aykırı değerler: \n{df_kopya.loc[sutun_maskesi, sutun]}")
    
print(f"En az bir aykırı değer içeren satırlar \n{df_kopya.loc[aykiri_deger_maskesi]}") 

sns.boxplot(x=df_kopya["math score"])
plt.show()

sns.boxplot(x=df_kopya["writing score"])
plt.show()

sns.histplot(x=df_kopya["math score"], kde=True)
plt.show()

label_encoder = LabelEncoder()
df_encoded = df_kopya.copy()

one_hot_sutunlar = [
    "gender",
    "race/ethnicity",
    "lunch",
    "test preparation course"
]

df_encoded = pd.get_dummies(
    df_encoded,
    columns=one_hot_sutunlar,
    dtype=int
)

print(df["parental level of education"].unique())
#[ 'bachelor's degree',       'some college',    'master's degree',
#'associate's degree',        'high school',   'some high school']

education_order = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree"
]

ordinal_encoder = OrdinalEncoder(
    categories=[education_order]
)

df_encoded[["parental level of education"]] = ordinal_encoder.fit_transform(
    df_encoded[["parental level of education"]]
)

print(df_encoded.head())
print(df_encoded.info())

X = df_encoded.drop(columns=["math score"])
y = df_encoded["math score"]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,test_size=0.25,random_state=42)

print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.mean())
print(X_train_scaled.std())

model = LinearRegression()

model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_val_scaled)

print(y_pred)

mae = mean_absolute_error(y_val, y_pred)

mse = mean_squared_error(y_val, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_val, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
"""
MAE: 4.355754154018615
MSE: 28.710893523919268
RMSE: 5.3582547087572525
R2 Score: 0.8673172884032205
"""
sonuclar = pd.DataFrame({
    "Gerçek Not": y_val.values,
    "Tahmin Edilen Not": y_pred
})

print(sonuclar.head(15))

plt.figure(figsize=(8, 6))

plt.scatter(y_val, y_pred)

plt.xlabel("Gerçek Matematik Notları")
plt.ylabel("Tahmin Edilen Matematik Notları")
plt.title("Gerçek vs Tahmin Edilen Notlar")

plt.show()

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_val)

mae = mean_absolute_error(y_val, rf_pred)
mse = mean_squared_error(y_val, rf_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, rf_pred)

print("Random Forest Sonuçları")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

korelasyonlar = df_encoded.corr(numeric_only=True)["math score"].sort_values(ascending=False)
print(korelasyonlar)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest - Feature Importance")
plt.show()

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi CV MAE:", -grid_search.best_score_)

best_rf_model = grid_search.best_estimator_

best_rf_pred = best_rf_model.predict(X_val)

mae = mean_absolute_error(y_val, best_rf_pred)
mse = mean_squared_error(y_val, best_rf_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, best_rf_pred)

print("Optimized Random Forest")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


y_test_pred = model.predict(X_test_scaled)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("FINAL TEST RESULTS")
print("MAE:", test_mae)
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R2 Score:", test_r2)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_test_pred, alpha=0.7)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--"
)

plt.xlabel("Gerçek Matematik Notları")
plt.ylabel("Tahmin Edilen Matematik Notları")
plt.title("Linear Regression - Actual vs Predicted")

plt.show()