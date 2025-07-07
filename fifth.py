import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import random

df = pd.read_csv("player_data.csv")
df["fpl_sel"] = df["fpl_sel"].str.rstrip('%').astype(float)
df = df.drop(columns=["name", "club", "nationality"])
df = pd.get_dummies(df, columns=["position", "age_cat", "club_id"], drop_first=True)
df = df.dropna(subset=["market_value"])
df = df.fillna(0)

X = df.drop(columns=["market_value"]).values
y = df["market_value"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X_train.shape[1]

def evaluate(weights):
    weights = np.clip(weights, 0, 1)
    X_train_w = X_train * weights
    X_test_w = X_test * weights
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_w, y_train)
    y_pred = knn.predict(X_test_w)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return -rmse  # higher fitness = better

def init_population(size):
    return [np.random.rand(n_features) for _ in range(size)]

def select(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(p1, p2, alpha=0.5):
    return alpha * p1 + (1 - alpha) * p2

def mutate(weights, mutation_rate=0.1, strength=0.2):
    for i in range(len(weights)):
        if random.random() < mutation_rate:
            weights[i] += np.random.uniform(-strength, strength)
    return np.clip(weights, 0, 1)

def genetic_algorithm(generations=50, population_size=30, mutation_boost=True):
    pop = init_population(population_size)
    best_fit = -np.inf
    stagnation = 0

    for gen in range(generations):
        fitnesses = [evaluate(ind) for ind in pop]
        new_pop = []

        if mutation_boost and stagnation >= 5:
            mutation_strength = 0.4  # boost mutation
        else:
            mutation_strength = 0.2

        for _ in range(population_size):
            p1 = select(pop, fitnesses)
            p2 = select(pop, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, strength=mutation_strength)
            new_pop.append(child)

        pop = new_pop
        gen_best = max(fitnesses)
        if gen_best > best_fit:
            best_fit = gen_best
            stagnation = 0
        else:
            stagnation += 1

        print(f"Generation {gen+1}: Best RMSE = {-best_fit:.4f}")

    best_weights = pop[np.argmax([evaluate(ind) for ind in pop])]
    return best_weights

best_weights = genetic_algorithm()

X_train_weighted = X_train * best_weights
X_test_weighted = X_test * best_weights
final_knn = KNeighborsRegressor(n_neighbors=5)
final_knn.fit(X_train_weighted, y_train)
y_pred = final_knn.predict(X_test_weighted)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n✅ Final RMSE with GA-optimized weights: {final_rmse:.4f}")
