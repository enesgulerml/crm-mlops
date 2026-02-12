import os
import sys
import pandas as pd
import gdown

# --- PATH SETTINGS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.common import load_config 

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        try:
            logger.info("Data ingestion is starting...")
            self.config = load_config()
            self.ingestion_config = self.config['data_ingestion']
        except Exception as e:
            logger.error("An error occurred while loading the configuration.")
            raise CustomException(e, sys)

    def download_data(self):
        try:
            dataset_path = self.ingestion_config['local_data_path']
            # Drive ID
            file_id = self.ingestion_config['drive_file_id']
            
            # Create folder
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

            if os.path.exists(dataset_path):
                logger.info(f"The file already exists: {dataset_path}")
                return dataset_path

            logger.info(f"Downloading from Google Drive... ID: {file_id}")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, dataset_path, quiet=False)
            
            logger.info(f"Download completed: {dataset_path}")
            return dataset_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            csv_path = self.download_data()
            
            logger.info("Reading data...")
            df = pd.read_csv(csv_path)
            
            if 'customerID' in df.columns:
                df = df.drop(['customerID'], axis=1)
                
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
            target_col = 'Churn'
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != target_col:
                    df[col] = df[col].fillna("Missing").astype(str)

            logger.info(f"Data Ingestion Successful! Size: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        df = obj.initiate_data_ingestion()
        print("\n----- SUCCESSFUL RESULT (HEAD) -----")
        print(df.head())
    except Exception as e:
        print(f"ERROR: {e}")