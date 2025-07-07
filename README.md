# MarketPredictorPlayer
#  Player Market Value Predictor

This is a machine learning web application that predicts the **market value** of English Premier League (EPL) players based on various performance, popularity, and club-related features. The app is built using **Streamlit** and trained on real 2016–17 player data.

## Features

- Predicts market value of EPL players using ML models
- Inputs include:
  - Age
  - Position category
  - FPL points, value, and selection %
  - Page views (popularity)
  - Club status (big club, new signing, etc.)
- Built using:
  - `GradientBoostingRegressor`
  - `scikit-learn`, `joblib`, `pandas`, `numpy`
- Clean and interactive frontend using Streamlit
- Deployed as a REST API and a web app

##  Input Features

| Feature          | Description                                      |
|------------------|--------------------------------------------------|
| `age`            | Player's age                                     |
| `position_cat`   | 1 = Attacker, 2 = Midfielder, 3 = Defender, 4 = GK|
| `page_views`     | Avg. daily Wikipedia page views                  |
| `fpl_value`      | FPL price                                        |
| `fpl_sel`        | % of FPL users who picked the player             |
| `fpl_points`     | FPL points earned last season                    |
| `region`         | 1 = England, 2 = EU, 3 = Americas, 4 = RoW        |
| `new_foreign`    | 1 = New signing from outside EPL                 |
| `big_club`       | 1 = Player belongs to a big club (Top 6)         |
| `new_signing`    | 1 = New signing for that season                  |


