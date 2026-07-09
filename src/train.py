"""
Trains the final Random Forest model using the
best hyperparameters identified during experimentation in Notebook,
evaluates its performance, and saves the trained model.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.config import MODEL_PATH

def train_model(x_train, y_train, Save:bool = True):
     """Fit the final Random Forest with the tuned hyperparameters."""
     model = RandomForestRegressor()
     model.fit(x_train, y_train)

     if Save:
          joblib.dump(model, MODEL_PATH)
     return model

    