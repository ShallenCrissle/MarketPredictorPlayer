import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

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

models = {
    "Ridge": Ridge(alpha=10),
    "Lasso": Lasso(alpha=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(C=10, epsilon=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
}

print("\n--- Model Selection Results ---")

best_model = None
best_score = -np.inf

for name, model in models.items():
    if name in ["SVR", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_mean = np.mean(cv_scores)

    print(f"{name}: CV R2 = {cv_mean:.3f}, Test R2 = {r2:.3f}, RMSE = {rmse:.2f}")

    if cv_mean > best_score:
        best_score = cv_mean
        best_model = name

print(f"\n✅ Best Model Based on CV R²: {best_model}")
