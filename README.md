# Quebec Startup Funding — Predicting Funding Round Size

**Author:** Eglantine Dupuy

An end-to-end machine learning project predicting the size of startup funding rounds in Quebec, using a regression model trained on Dealroom.co data. Built as Assignment 1 for Enterprise Data Science.

## Project overview

- **Business objective:** Understand what predicts the size of a startup funding round in Quebec, as a benchmarking tool for founders/investors comparing a prospective round to similar Quebec companies — not as a standalone decision-making tool.
- **Framing:** Supervised learning, regression, on a fixed historical dataset of funding rounds (not a live/streaming problem).
- **Target variable:** `Amount (USD M)` — the disclosed size of a funding round.
- **Performance measure:** RMSE (primary), with MAE as a secondary metric, since funding amounts are heavily right-skewed by a handful of mega-rounds.
- **Population:** Quebec-HQ'd companies with funding rounds dated 2021–present (data exported 16/09/2026).

## Data

- **Source:** [Dealroom.co](https://dealroom.co) — a global data platform on tech ecosystems, startups, and venture capital (premium/paid access required).
- The raw data export is **not included in this repo** (it is proprietary, licensed data from Dealroom), so the notebook will not run end-to-end without access to an equivalent export. The notebook itself documents every step — data cleaning, exploration, feature engineering, modeling, and evaluation — so it can be read and followed without re-running it.

## Methodology summary

1. **Data preparation:** Filtered to Quebec-HQ'd companies, dropped rows with an undisclosed round amount (~45.6%, concentrated in early-stage/undisclosed round types — an explicit limitation), engineered `Government Backed` / `Semi-Public` / `Publicly Backed` flags for public and quasi-public investor participation.
2. **Exploration:** Examined disclosure patterns by round type, the distribution of round sizes, and how round amount varies with public/semi-public backing.
3. **Preprocessing:** Median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features, via a `ColumnTransformer` pipeline.
4. **Modeling:** Compared Linear Regression, Decision Tree, and Random Forest using 5-fold cross-validation. Random Forest performed best and was selected for tuning.
5. **Fine-tuning & evaluation:** Random forest hyperparameters tuned with `RandomizedSearchCV`; final model evaluated on a held-out test set (RMSE ≈ $19.3M, MAE ≈ $7.6M), with a bootstrap 95% confidence interval for test RMSE.
6. **Feature importance:** Growth stage, launch year, total rounds raised, and round type were the strongest predictors. `Publicly Backed` showed a clear association with larger rounds in simple comparisons but ranked low in feature importance — suggesting the association is largely explained by publicly backed rounds occurring at later funding stages, rather than public backing independently driving round size.

## Key limitations

- Substantial, non-random target missingness (~46%) biases the usable sample toward later-stage, better-documented rounds.
- Right-skewed target with a few very large rounds, and a modest sample size (~621 disclosed rounds) given that skew.
- Growth stage and total rounds number are measured as of the data export, not at the time of each round — a potential source of look-ahead leakage.
- Industry tags were multi-label strings, which produced many sparse one-hot columns rather than a clean, useful encoding.
- Associations found (e.g. public backing) are not evidence of a causal effect — there is no random assignment or control region in this design.

## Tech stack

Python, pandas, NumPy, matplotlib, scikit-learn (`Pipeline`, `ColumnTransformer`, `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `RandomizedSearchCV`), SciPy, joblib.

## Acknowledgements

Structure and several helper functions adapted from Aurélien Géron's end-to-end ML project notebook in *Hands-On Machine Learning* (referred to as the "reference notebook" throughout).

## Contents

- [`EglantineDupuy_EDS_Assignment1.ipynb`](./EglantineDupuy_EDS_Assignment1.ipynb) — the full analysis notebook.
