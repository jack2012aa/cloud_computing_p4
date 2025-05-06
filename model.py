import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # A relatively simple and fast model
# from xgboost import XGBRegressor # If you want to try XGBoost
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import os

# --- 1. Configuration ---
DATA_PATH = './data/'  # Assume train.csv and test.csv are in the 'data' subdirectory
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test.csv')
SUBMISSION_FILE_LOCAL = 'submission_local_test.csv'

# Use a subset of data for fast local testing; set to None to use the full dataset
# For housing price dataset, local processing is relatively fast even with full dataset
SAMPLE_SIZE = 1000  # For example, first test with 1000 samples
# SAMPLE_SIZE = None # Use the full dataset

RANDOM_STATE = 42  # To ensure reproducibility

# --- 2. Helper Functions ---

def load_data(train_path, test_path, sample_size=None):
    """Load training and test data"""
    print(f"Loading data... Sample size: {'full' if sample_size is None else sample_size}")
    try:
        if sample_size:
            train_df = pd.read_csv(train_path, nrows=sample_size)
            # For local testing, to ensure test_df has the same columns as train_df (since sample_size might not include all categories)
            # For actual Kaggle submission, test_df should be fully read
            if os.path.exists(test_path):  # Ensure test_path exists
                 test_df = pd.read_csv(test_path, nrows=sample_size)  # Sample test data for local testing
            else:
                print(f"Warning: Test file {test_path} not found. Creating an empty test_df.")
                test_df = pd.DataFrame()  # Create an empty DataFrame if test.csv is missing
        else:
            train_df = pd.read_csv(train_path)
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
            else:
                print(f"Warning: Test file {test_path} not found. Creating an empty test_df.")
                test_df = pd.DataFrame()

        print(f"Training data shape: {train_df.shape}")
        if not test_df.empty:
            print(f"Test data shape: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data file is located in the '{DATA_PATH}' directory.")
        return pd.DataFrame(), pd.DataFrame()

def preprocess_data(train_df, test_df):
    """Perform basic data preprocessing"""
    print("Starting data preprocessing...")

    # 1. Store Id and process target variable
    train_ids = train_df['Id']
    if not test_df.empty:
        test_ids = test_df['Id']
    else:
        test_ids = pd.Series(dtype='int')  # If test_df is empty

    y_train_log = np.log1p(train_df['SalePrice'])  # Apply log transformation to target variable

    # 2. Remove Id and target variable, then merge feature sets for consistent processing
    X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
    if not test_df.empty:
        X_test = test_df.drop('Id', axis=1)
        print(f"Before merging - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        all_features = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    else:
        print(f"Before merging - X_train shape: {X_train.shape}, X_test is empty")
        all_features = X_train.copy().reset_index(drop=True)

    print(f"Merged features shape: {all_features.shape}")

    # 3. Identify numerical and categorical columns
    numerical_cols = all_features.select_dtypes(include=np.number).columns
    categorical_cols = all_features.select_dtypes(include='object').columns

    # 4. Handle missing values
    # Numerical: fill missing values with median
    num_imputer = SimpleImputer(strategy='median')
    all_features[numerical_cols] = num_imputer.fit_transform(all_features[numerical_cols])

    # Categorical: fill missing values with 'Missing' (or mode)
    for col in categorical_cols:
        all_features[col] = all_features[col].fillna('Missing')

    # 5. One-Hot Encoding for categorical features
    all_features = pd.get_dummies(all_features, columns=categorical_cols, dummy_na=False)
    print(f"Shape after one-hot encoding: {all_features.shape}")

    # 6. Separate back into training and test sets
    X_train_processed = all_features.iloc[:len(X_train), :]
    if not test_df.empty:
        X_test_processed = all_features.iloc[len(X_train):, :]
    else:
        X_test_processed = pd.DataFrame(columns=X_train_processed.columns)  # Create an empty DataFrame with the same columns if test_df is empty

    print(f"Processed X_train shape: {X_train_processed.shape}")
    if not test_df.empty:
        print(f"Processed X_test shape: {X_test_processed.shape}")

    return X_train_processed, y_train_log, X_test_processed, test_ids

def train_and_evaluate_model(X_train, y_train, random_state=RANDOM_STATE):
    """Split a portion of the local training set for validation and train a model"""
    print("Splitting data for local validation...")
    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    print(f"Starting local model training (Number of training samples: {X_train_part.shape[0]})...")
    # Choose a model, RandomForest is fast; XGBoost might perform better but requires installation
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    # model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=random_state, objective='reg:squarederror', n_jobs=-1)

    model.fit(X_train_part, y_train_part)
    print("Model training complete.")

    # Evaluate on the local validation set
    y_pred_val_log = model.predict(X_val_part)
    rmse = np.sqrt(mean_squared_error(y_val_part, y_pred_val_log))
    print(f"Local validation RMSE (based on log-transformed SalePrice): {rmse:.4f}")
    return model

def generate_submission(model, X_test, test_ids, submission_file):
    """Generate predictions and save as submission file"""
    if X_test.empty or test_ids.empty:
        print("Test data is empty; cannot generate submission file.")
        return pd.DataFrame()

    print(f"Predicting on {X_test.shape[0]} test samples...")
    log_predictions = model.predict(X_test)
    final_predictions = np.expm1(log_predictions)  # Reverse log transformation

    submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file '{submission_file}' successfully created.")
    return submission_df

# --- 3. Main Execution Flow ---
if __name__ == "__main__":
    # Check if the data directory exists; if not, create it
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Data directory '{DATA_PATH}' created.")

    # 1. Load data
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE, sample_size=SAMPLE_SIZE)

    if not train_df.empty:  # Training data is required
        # 2. Data preprocessing
        # Use .copy() to avoid SettingWithCopyWarning, though it might not be an issue in this flow
        X_train_p, y_train_log_p, X_test_p, test_ids_p = preprocess_data(train_df.copy(), test_df.copy())

        # 3. Train and evaluate the model
        model = train_and_evaluate_model(X_train_p, y_train_log_p, random_state=RANDOM_STATE)

        # 4. Generate submission file (if test data exists)
        if not X_test_p.empty:
            submission = generate_submission(model, X_test_p, test_ids_p, SUBMISSION_FILE_LOCAL)
            if not submission.empty:
                 print("\nLocal test script execution is complete.")
                 print(f"First few rows of prediction file '{SUBMISSION_FILE_LOCAL}':")
                 print(submission.head())
        else:
            print("\nLocal test script execution complete (no test data for prediction).")
    else:
        print("Cannot execute because training data failed to load.")