import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

df = pd.read_csv("player_data.csv")

df["fpl_sel"] = df["fpl_sel"].str.rstrip('%').astype(float)
df = df.drop(columns=["name", "club", "nationality"])
df = pd.get_dummies(df, columns=["position", "age_cat", "club_id"], drop_first=True)
df = df.dropna(subset=["market_value"])
df = df.fillna(0)

X = df.drop(columns=["market_value"])
y = df["market_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter grids
param_grids = {
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0]
    },
    "Lasso": {
        "alpha": [0.01, 0.1, 1.0, 10.0]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9]
    },
    "SVR": {
        "C": [0.1, 1, 10],
        "epsilon": [0.1, 0.2, 0.5],
        "kernel": ["rbf"]
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5]
    }
}

models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

best_models = {}
print("\n--- Hyperparameter Tuning Results ---")

for name, model in models.items():
    params = param_grids[name]
    if name in ["SVR", "KNN"]:
        search = GridSearchCV(model, params, cv=5, scoring="r2", n_jobs=-1)
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
    else:
        search = GridSearchCV(model, params, cv=5, scoring="r2", n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    best_models[name] = (best_model, rmse, r2)
    print(f"{name}: Best Params = {search.best_params_}, RMSE = {rmse:.2f}, R2 = {r2:.2f}")
import joblib

joblib.dump(best_model, "gb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
