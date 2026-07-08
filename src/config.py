'''
Place for Storing file PATH, CONSTANTS and HYPERPARAMETERS
'''

# --- Paths ---
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR /'data' /'raw'/ 'temp.csv'
PROCESSED_DATA_PATH = ROOT_DIR /'data' /'processed'/ 'cleaned_data.csv'
MODEL_PATH = ROOT_DIR /'models' / 'model.pkl'
SCALER_PATH = ROOT_DIR /'models' / 'scaler.pkl'

# --- Target Variable ---
TARGET_VARIABLE = ['Next_Tmax', 'Next_Tmin']

# --- HyperParameter for GridSearchCV
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}