"""
Loads the raw LDAPS dataset and applies the same cleaning / feature
engineering steps used in the original exploration notebook:
  - fill missing numeric values with the median
  - forward-fill missing dates
  - break the Date column into year / month / day / day-of-year
  - drop the now-redundant Date column
"""

import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# --- Import Data ---
def load_raw_data(path = RAW_DATA_PATH) -> pd.DataFrame:
    ''' Read The Raw CSV '''
    if not path.exists:
        raise FileNotFoundError(
            f" Couldn't Found The DataSet at {path}"
            f" Download The File at https://www.kaggle.com/datasets/smokingkrils/temperature-forecast-project-using-ml"
            f" And Plase It There"
        )
    return pd.read_csv(path)

# --- Split Date Column and Fill NA ---
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert to Datetime
    df['Date'] = pd.to_datetime(df['Date'])

    #fill NA
    df = df.fillna(df.median(numeric_only=True))
    df['Date'] = df['Date'].ffill()

    # Break down the Date column
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Drop Date column
    df.drop('Date', axis= 1, inplace= True )
    return df

# --- Load and Prepare ---
def load_prepare(Save: bool= True) -> pd.DataFrame:

    df = load_raw_data()
    df = feature_engineer(df)

    if Save:
        df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df

if __name__ == "__main__":
    data = load_prepare()
    print(f"Loaded and Processed {data.shape[0]} Raws, {data.shape[1]} Columns")
