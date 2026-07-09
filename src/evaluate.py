"""
Final evaluation of the Performence of the Tuned Model on the held-out test set.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)

    tests = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "MSE": round(mean_squared_error(y_test, y_pred), 2),
        "R2": round(r2_score(y_test, y_pred), 2),
    }
    print("=" * 50)
    print("Final Model Evaluation")
    print("=" * 50)
    for key, value in tests.items():
        print(f"{key} : {value}")

    return tests