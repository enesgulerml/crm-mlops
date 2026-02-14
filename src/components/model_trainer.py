import sys
import os
import xgboost as xgb
import optuna
import mlflow
import mlflow.onnx
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple
from dotenv import load_dotenv # Şifreleri okumak için

# Path Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.common import load_config

# Initialize Logger
logger = get_logger(__name__)

# .env dosyasını yükle (Şifreler sisteme tanımlanır)
load_dotenv()

class ModelTrainer:
    def __init__(self):
        try:
            logger.info("Initializing Model Trainer configuration...")
            self.config = load_config()
            self.trainer_config = self.config['model_trainer']
            self.n_trials = self.trainer_config['n_trials']
            self.onnx_path = self.trainer_config['output_path']
            
            # MLflow Bağlantısı (.env'den okur)
            logger.info("Connecting to MLflow Tracking Server...")
            
            # Eğer .env yoksa veya okuyamazsa hata vermesin diye kontrol
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment("CRM_Churn_Prediction")
                logger.info(f"MLflow connected: {tracking_uri}")
            else:
                logger.warning("MLFLOW_TRACKING_URI not found in environment variables!")

        except Exception as e:
            logger.error("Failed to initialize Model Trainer.")
            raise CustomException(e, sys)

    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test) -> dict:
            def objective(trial):
                # Her bir deneme (trial) için yeni bir "Nested Run" başlatıyoruz
                with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                    param = {
                        'verbosity': 0,
                        'objective': 'binary:logistic',
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'random_state': 42
                    }
                    
                    model = xgb.XGBClassifier(**param)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    accuracy = accuracy_score(y_test, preds)
                    
                    # Her denemenin parametrelerini ve sonucunu logla
                    mlflow.log_params(param)
                    mlflow.log_metric("accuracy", accuracy)
                    
                    return accuracy

            logger.info(f"Starting nested hyperparameter optimization...")
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials) # config'deki n_trials kadar döner
            
            return study.best_params

    def train_and_save_onnx(self, X_train, y_train, X_test, y_test):
        try:
            # MLflow Run Başlat
            with mlflow.start_run():
                
                # 1. Optimizasyon
                best_params = self.optimize_hyperparameters(X_train, y_train, X_test, y_test)
                
                # MLflow'a Parametreleri Yaz 📝
                mlflow.log_params(best_params)

                # 2. Final Model Eğitimi
                logger.info("Training Final Model...")
                final_model = xgb.XGBClassifier(**best_params)
                final_model.fit(X_train, y_train)
                
                # 3. Metrikleri Hesapla
                y_pred = final_model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # MLflow'a Metrikleri Yaz 📊
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                
                logger.info(f"Metrics -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

                # 4. ONNX Dönüşümü
                logger.info("Converting to ONNX...")
                initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
                onnx_model = onnxmltools.convert_xgboost(final_model, initial_types=initial_type)
                
                # Yerel Kayıt
                os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
                onnxmltools.utils.save_model(onnx_model, self.onnx_path)
                
                # MLflow'a Modeli Yükle (Artifact) 📦
                # Hem ONNX hem de XGBoost formatında saklayalım
                mlflow.onnx.log_model(onnx_model, "model_onnx")
                mlflow.xgboost.log_model(final_model, "model_xgboost")
                
                logger.info("✅ Training Completed & Logged to MLflow!")
                
        except Exception as e:
            logger.error("Error in training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    try:
        logger.info(">>>>> MLflow Execution Started <<<<<")
        ingestion = DataIngestion()
        df = ingestion.initiate_data_ingestion()
        
        transformer = DataTransformation()
        X_train, X_test, y_train, y_test, _ = transformer.initiate_data_transformation(df)
        
        trainer = ModelTrainer()
        trainer.train_and_save_onnx(X_train, y_train, X_test, y_test)
        logger.info(">>>>> Execution Finished <<<<<")
        
    except Exception as e:
        logger.error(e)
        print(e)