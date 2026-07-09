# Next-Day Temperature Forecasting (LDAPS Bias Correction)

Predicts next-day maximum and minimum temperature (`Next_Tmax`, `Next_Tmin`) using the UCI/Kaggle LDAPS temperature forecast dataset, treating it as a regression bias-correction problem on top of numerical weather prediction outputs.

## Problem Statement

Numerical weather models (LDAPS) produce raw forecasts that carry systematic bias. This project trains a machine learning model on historical LDAPS features plus station/date metadata to correct that bias and predict the next day's max/min temperature more accurately.

## Dataset

- **Source:** [Temperature Forecast Project Using ML](https://www.kaggle.com/datasets/smokingkrils/temperature-forecast-project-using-ml) (Kaggle mirror of the UCI LDAPS dataset)
- **Targets:** `Next_Tmax`, `Next_Tmin`
- **Features:** LDAPS model outputs (cloud cover, humidity, etc.), station ID, and date-derived features (year, month, day, day-of-year)

Place the raw `temp.csv` file in `data/raw/` before running the pipeline (not included in this repo — see `.gitignore`).

## Project Structure

```
weather-temp-prediction/
├── data/
│   ├── raw/              # place temp.csv here
│   └── processed/        # cleaned data cached here after running the pipeline
├── notebooks/
│   └── exploration.ipynb # original EDA & experimentation notebook
├── src/
│   ├── config.py         # paths, constants, hyperparameter grid
│   ├── data_loader.py    # loading + cleaning + feature engineering
│   ├── preprocessing.py  # train/test split, one-hot encoding, scaling
│   ├── train.py          # best fit model training
│   └── evaluate.py       # final metrics reporting
├── models/                # saved model_final.pkl and scaler.pkl (generated)
├── main.py                # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Pipeline Overview

1. **Load & clean** — read raw CSV, fill missing values with median, engineer date features (year/month/day/day-of-year), drop the raw date column.
2. **Preprocess** — one-hot encode `station`, split into train/test (80/20), scale features with `StandardScaler` (fit on train only, to avoid leakage into the test set).
3. **Training** — GridSearchCV over Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`), since it performed best.
4. **Final evaluation** — fit the tuned Random Forest and report final test-set metrics.

## Setup

```bash
git clone <your-repo-url>
cd weather-temp-prediction
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Place `temp.csv` in `data/raw/`, then run:

```bash
python main.py
```

This prints model comparison tables, cross-validation results, the best GridSearchCV hyperparameters, and final test-set metrics — and saves the trained model to `models/model_final.pkl`.


# Random Forest was selected as the final model based on cross-validation and test-set performance.

## Author

Built by [Pranam](https://github.com/Pranamchand) as part of an ongoing data science / ML engineering portfolio.
