"""
End-to-end pipeline for the Next-Day Tmax/Tmin forecasting model.

Pipeline:
1. Load dataset
2. Preprocess data
3. Train model
4. Evaluate model

Run With python main.py ---
"""

import pandas as pd
from src.data_loader import load_prepare
from src.preprocessing import prepeare_train_test
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("=" * 60)
    print(" Temperature Prediction Pipeline Started")
    print("=" * 60)


    # -------------------------------------------------
    # Step 1 : Load Dataset
    # -------------------------------------------------
    print("\n[1/4] Loading dataset...")
    df = load_prepare()
    print(f"Dataset loaded successfully.")
    print(f"{df.shape[0]} Raws, {df.shape[1]} Columns")


    # -------------------------------------------------
    # Step 2 : Data Preprocessing
    # -------------------------------------------------
    print("\n[2/4] Preprocessing data...")
    x_train, x_test, y_train, y_test, scaler = prepeare_train_test(df)
    print("Data preprocessing completed.")
    print(f"Training samples : {x_train.shape[0]}")
    print(f"Testing samples  : {x_test.shape[0]}")


    # -------------------------------------------------
    # Step 3 : Train Model
    # -------------------------------------------------
    print("\n[3/4] Training Random Forest model...")
    model = train_model(x_train, y_train)
    print("Model training completed.")


    # -------------------------------------------------
    # Step 4 : Evaluate Model
    # -------------------------------------------------
    print("\n[4/4] Evaluating model...")
    evaluate_model(model, x_test, y_test)
    print("\nEvaluation completed.")
    print("\n" + "=" * 60)
    print(" Temperature Prediction Pipeline Finished")
    print("=" * 60)


if __name__ == "__main__":
    main()