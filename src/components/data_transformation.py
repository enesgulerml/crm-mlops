import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd

# Path Settings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# My own modules
from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.common import load_config, save_object

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        try:
            self.config = load_config()
            # I'm retrieving file paths from the config file.
            self.preprocessor_obj_file_path = self.config['data_transformation']['preprocessor_obj_file_path']
            self.test_size = self.config['base']['test_size']
            self.random_state = self.config['base']['random_state']
            self.target_col = 'Churn'  # Target Value
        except Exception as e:
            logger.error("An error occurred while loading the configuration.")
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        It creates the data transformation pipeline (rules).
        """
        try:
            # Numerical Variables
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
            
            # Categorical Variables (excluding Target 'Churn')
            categorical_columns = [
                "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
                "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
                "Contract", "PaperlessBilling", "PaymentMethod"
            ]

            # 1. Numerical Pipeline: Fill in the blanjs with the median -> Standardize
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # 2. Categorical Pipeline: 'Miss' empty spaces -> OneHot Encode -> Standardize
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logger.info(f"Numerical Columns: {numerical_columns}")
            logger.info(f"Categorical Columns: {categorical_columns}")

            # ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, df):
        try:
            logger.info("Data Transformation has been launched.")
            
            # --- Manual Adjustments ---
            # TotalCharges can sometimes be a string, convert it to a number.
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
            # Convert Targegt (Chrun) to a number (Yes->1, No->0)
            logger.info("The target variable (Churn) is being encoded....")
            df[self.target_col] = df[self.target_col].map({'Yes': 1, 'No': 0})
            
            # Sperate x and y.
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]

            # --- PREPROCESSOR ---
            preprocessing_obj = self.get_data_transformer_object()

            # --- TRAIN-TEST SPLIT ---
            # The prevent data leakage, I first divide it.
            logger.info("The data is divided into Train-Test sections...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # --- FIT & TRANSFORM ---
            logger.info("Preprocessing is being applied...")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            # --- SAVING (PICKLE) ---
            logger.info(f"Preprocesser is being registered: {self.preprocessor_obj_file_path}")
            save_object(
                file_path=self.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logger.info("Data transformation has been successfully completed.")
            
            return (
                X_train_arr,
                X_test_arr,
                y_train,
                y_test,
                self.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    try:
        ingestion = DataIngestion()
        df = ingestion.initiate_data_ingestion()
        
        transformer = DataTransformation()
        X_train, X_test, y_train, y_test, _ = transformer.initiate_data_transformation(df)
        
        print("\n----- TRANSFORMATION RESULT -----")
        print(f"x_train size: {X_train.shape}")
        print(f"x_test size: {X_test.shape}")
        print(f"y_train size: {y_train.shape}")
        print("The preprocessor object was saved as 'models/preprocessor.pkl'.")
        
    except Exception as e:
        logger.error(e)
        print(f"ERROR: {e}")