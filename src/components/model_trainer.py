import sys
import os
import xgboost as xgb
import optuna
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, Any

# Path Configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.common import load_config

# Initialize Logger
logger = get_logger(__name__)

class ModelTrainer:
    """
    Handles the model training pipeline including hyperparameter optimization via Optuna
    and model serialization to ONNX format.
    """
    def __init__(self):
        try:
            logger.info("Initializing Model Trainer configuration...")
            self.config = load_config()
            self.trainer_config = self.config['model_trainer']
            self.n_trials = self.trainer_config['n_trials']
            self.onnx_path = self.trainer_config['output_path']
        except Exception as e:
            logger.error("Failed to initialize Model Trainer configuration.")
            raise CustomException(e, sys)

    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test) -> dict:
        """
        Optimizes XGBoost hyperparameters using Optuna.

        Returns:
            dict: Best hyperparameters found by Optuna.
        """
        def objective(trial):
            param = {
                'verbosity': 0,
                'objective': 'binary:logistic',
                'booster': 'gbtree',
                'tree_method': 'hist',  # Efficient for large datasets
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42
            }

            model = xgb.XGBClassifier(**param)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            return accuracy

        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        logger.info(f"Optimization completed. Best parameters: {study.best_params}")
        return study.best_params

    def train_and_save_onnx(self, X_train, y_train, X_test, y_test) -> Tuple[float, float]:
        """
        Trains the final model using the best hyperparameters and saves it as an ONNX file.

        Returns:
            Tuple[float, float]: Accuracy and F1 Score of the trained model.
        """
        try:
            # 1. Hyperparameter Optimization
            best_params = self.optimize_hyperparameters(X_train, y_train, X_test, y_test)
            
            # 2. Train Final Model
            logger.info("Training the final model with optimal parameters...")
            final_model = xgb.XGBClassifier(**best_params)
            final_model.fit(X_train, y_train)
            
            # 3. Evaluate Performance
            y_pred = final_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            logger.info(f"Model Evaluation -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

            # 4. Convert to ONNX
            logger.info("Converting model to ONNX format...")
            
            # Define input type (Float Tensor with dynamic batch size)
            initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
            
            onnx_model = onnxmltools.convert_xgboost(final_model, initial_types=initial_type)
            
            # 5. Save Model
            os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
            onnxmltools.utils.save_model(onnx_model, self.onnx_path)
            
            logger.info(f"✅ Model successfully saved at: {self.onnx_path}")
            return acc, f1

        except Exception as e:
            logger.error("Error occurred during model training/saving process.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # --- ISOLATED TEST BLOCK ---
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    try:
        logger.info(">>>>> STAGE: Model Training Started <<<<<")
        
        # 1. Ingestion
        ingestion = DataIngestion()
        df = ingestion.initiate_data_ingestion()
        
        # 2. Transformation
        transformer = DataTransformation()
        X_train, X_test, y_train, y_test, _ = transformer.initiate_data_transformation(df)
        
        # 3. Training
        trainer = ModelTrainer()
        trainer.train_and_save_onnx(X_train, y_train, X_test, y_test)
        
        logger.info(">>>>> STAGE: Model Training Completed Successfully <<<<<")
        
    except Exception as e:
        logger.error(e)
        print(f"❌ CRITICAL ERROR: {e}")