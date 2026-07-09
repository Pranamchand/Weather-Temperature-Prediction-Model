"""
Splits features/targets, one-hot encodes the station column,
and scales features with StandardScaler.

"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_VARIABLE, SCALER_PATH

# --- Split the Data into x and y and encode station column ---
def split_target_feature(df :pd.DataFrame):
     x = df.drop(columns=TARGET_VARIABLE)
     y = df[TARGET_VARIABLE]

     x["station"] = x["station"].astype("category")
     x = pd.get_dummies(x, columns=["station"], drop_first=False)
     return x, y

# --- Train Test and Split ---
def split_train_test(x, y):
    return train_test_split(
        x, y, test_size=0.2, random_state=42
    )

# --- Standerd Scaling ---
def scale_features(x_test, x_train, Save:bool = True):
     scaler = StandardScaler()

     x_train_scaled = scaler.fit_transform(x_train)
     x_test_scaled = scaler.transform(x_test)

     if Save:
          joblib.dump(scaler, SCALER_PATH)

     return x_test_scaled, x_train_scaled, scaler

# --- Final Function Combining all The Processes
def prepeare_train_test(df: pd.Dataframe):
     '''  encode -> split -> scale '''
     x, y = split_target_feature(df)
     x_train, x_test, y_train, y_test = split_train_test(x, y)
     x_test_scaled, x_train_scaled, scaler = scale_features(x_test, x_train)

     return x_train_scaled, x_test_scaled, y_train, y_test, scaler

